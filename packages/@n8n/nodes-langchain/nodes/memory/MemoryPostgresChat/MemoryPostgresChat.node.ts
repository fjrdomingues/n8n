/* eslint-disable n8n-nodes-base/node-dirname-against-convention */
import { PostgresChatMessageHistory } from '@langchain/community/stores/message/postgres';
import { BufferMemory, BufferWindowMemory } from 'langchain/memory';
import { BaseMemory } from '@langchain/core/memory';
import { BaseMessage, HumanMessage, AIMessage } from '@langchain/core/messages';
import { configurePostgres } from 'n8n-nodes-base/dist/nodes/Postgres/transport/index';
import type { PostgresNodeCredentials } from 'n8n-nodes-base/dist/nodes/Postgres/v2/helpers/interfaces';
import { postgresConnectionTest } from 'n8n-nodes-base/dist/nodes/Postgres/v2/methods/credentialTest';
import type {
	ISupplyDataFunctions,
	INodeType,
	INodeTypeDescription,
	SupplyData,
	NodeOperationError,
} from 'n8n-workflow';
import { NodeConnectionTypes } from 'n8n-workflow';
import type pg from 'pg';

import { getSessionId } from '@utils/helpers';
import { logWrapper } from '@utils/logWrapper';
import { getConnectionHintNoticeField } from '@utils/sharedFields';

import {
	sessionIdOption,
	sessionKeyProperty,
	contextWindowLengthProperty,
	expressionSessionKeyProperty,
} from '../descriptions';

// PostgreSQL-based tool aware memory implementation
class PostgresToolAwareMemory extends BaseMemory {
	public history: PostgresChatMessageHistory;
	public memoryKey: string = 'chat_history'; // Default memory key
	public inputKey: string | undefined = 'input';
	public outputKey: string | undefined = 'output';
	public returnMessages: boolean = true;
	public chatHistory: PostgresChatMessageHistory;

	constructor(input?: {
		memoryKey?: string;
		inputKey?: string;
		outputKey?: string;
		chatHistory: PostgresChatMessageHistory;
		returnMessages?: boolean;
	}) {
		super();

		this.returnMessages = input?.returnMessages ?? true;
		this.memoryKey = input?.memoryKey ?? 'chat_history';

		// The history field must be a valid PostgresChatMessageHistory instance
		if (!input?.chatHistory) {
			throw new Error('PostgresChatMessageHistory instance is required for PostgresToolAwareMemory');
		}
		this.history = input.chatHistory;
		this.chatHistory = this.history; // For BaseMemory compatibility
		this.inputKey = input?.inputKey ?? this.inputKey;
		this.outputKey = input?.outputKey ?? this.outputKey;
	}

	get memoryKeys(): string[] {
		// Return keys used by this memory class
		return [this.memoryKey];
	}

	async loadMemoryVariables(_values: Record<string, any>): Promise<Record<string, any>> {
		const messages = await this.history.getMessages();
		// Ensure the key matches what the agent expects (usually 'chat_history')
		return { [this.memoryKey]: messages };
	}

	async saveContext(
		inputs: Record<string, any>,
		outputs: Record<string, any>,
	): Promise<void> {
		// 1. Add the input message (assuming it's under the inputKey)
		if (this.inputKey && inputs[this.inputKey]) {
			await this.history.addMessage(new HumanMessage(inputs[this.inputKey]));
		}

		// 2. Add the output message(s) (assuming under outputKey)
		if (this.outputKey && outputs[this.outputKey]) {
			const outputValue = outputs[this.outputKey];

			if (typeof outputValue === 'string') {
				// If output is a simple string, wrap it in AIMessage
				await this.history.addMessage(new AIMessage(outputValue));
			} else if (outputValue instanceof AIMessage) {
				// If output is already an AIMessage (potentially with tool_calls), add it directly
				// This is CRUCIAL for preserving tool_calls
				await this.history.addMessage(outputValue);
			} else if (Array.isArray(outputValue) && outputValue.every((msg: any) => msg instanceof BaseMessage)) {
				// If output is an array of BaseMessages (e.g., AIMessage followed by ToolMessage)
				await this.history.addMessages(outputValue);
			} else {
				// Handle other types by converting to string
				await this.history.addMessage(new AIMessage(JSON.stringify(outputValue)));
			}
		}
	}

	async clear(): Promise<void> {
		await this.history.clear();
	}
}

export class MemoryPostgresChat implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Postgres Chat Memory',
		name: 'memoryPostgresChat',
		icon: 'file:postgres.svg',
		group: ['transform'],
		version: [1, 1.1, 1.2, 1.3, 1.4],
		description: 'Stores the chat history in Postgres table.',
		defaults: {
			name: 'Postgres Chat Memory',
		},
		credentials: [
			{
				name: 'postgres',
				required: true,
				testedBy: 'postgresConnectionTest',
			},
		],
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Memory'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.memorypostgreschat/',
					},
				],
			},
		},
		// eslint-disable-next-line n8n-nodes-base/node-class-description-inputs-wrong-regular-node
		inputs: [],
		// eslint-disable-next-line n8n-nodes-base/node-class-description-outputs-wrong
		outputs: [NodeConnectionTypes.AiMemory],
		outputNames: ['Memory'],
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiAgent]),
			sessionIdOption,
			expressionSessionKeyProperty(1.2),
			sessionKeyProperty,
			{
				displayName: 'Table Name',
				name: 'tableName',
				type: 'string',
				default: 'n8n_chat_histories',
				description:
					'The table name to store the chat history in. If table does not exist, it will be created.',
			},
			{
				...contextWindowLengthProperty,
				displayOptions: { hide: { '@version': [{ _cnd: { lt: 1.1 } }] } },
			},
			{
				displayName: 'Support Tool Calls',
				name: 'supportToolCalls',
				type: 'boolean',
				default: false,
				description: 'Whether to support tool/function calls in memory',
				displayOptions: { show: { '@version': [{ _cnd: { gte: 1.4 } }] } },
			},
		],
	};

	methods = {
		credentialTest: {
			postgresConnectionTest,
		},
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials<PostgresNodeCredentials>('postgres');
		const tableName = this.getNodeParameter('tableName', itemIndex, 'n8n_chat_histories') as string;
		const sessionId = getSessionId(this, itemIndex);
		const supportToolCalls = this.getNodeParameter('supportToolCalls', itemIndex, false) as boolean;

		const pgConf = await configurePostgres.call(this, credentials);
		const pool = pgConf.db.$pool as unknown as pg.Pool;

		const pgChatHistory = new PostgresChatMessageHistory({
			pool,
			sessionId,
			tableName,
		});

		// For version 1.4 and above with tool calls support enabled, use PostgresToolAwareMemory
		if (this.getNode().typeVersion >= 1.4 && supportToolCalls) {
			const memory = new PostgresToolAwareMemory({
				memoryKey: 'chat_history',
				chatHistory: pgChatHistory,
				returnMessages: true,
				inputKey: 'input',
				outputKey: 'output',
			});

			return {
				response: logWrapper(memory, this),
			};
		}

		// For older versions or when tool calls support is disabled, use the original implementation
		const memClass = this.getNode().typeVersion < 1.1 ? BufferMemory : BufferWindowMemory;
		const kOptions =
			this.getNode().typeVersion < 1.1
				? {}
				: { k: this.getNodeParameter('contextWindowLength', itemIndex) };

		const memory = new memClass({
			memoryKey: 'chat_history',
			chatHistory: pgChatHistory,
			returnMessages: true,
			inputKey: 'input',
			outputKey: 'output',
			...kOptions,
		});

		return {
			response: logWrapper(memory, this),
		};
	}
}
