import { describe, expect, it } from "vitest";
import { convertMessages } from "../src/providers/openai-completions.ts";
import { convertResponsesMessages } from "../src/providers/openai-responses-shared.ts";
import type { AssistantMessage, Context, Model, ToolResultMessage, Usage } from "../src/types.ts";

const usage: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const completionsModel: Model<"openai-completions"> = {
	id: "test-model",
	name: "Test OpenAI-compatible chat model",
	api: "openai-completions",
	provider: "openai",
	baseUrl: "https://api.openai.com/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 4096,
};

const responsesModel: Model<"openai-responses"> = {
	...completionsModel,
	api: "openai-responses",
	name: "Test OpenAI-compatible responses model",
};

const completionsCompat: Parameters<typeof convertMessages>[2] = {
	supportsStore: true,
	supportsDeveloperRole: true,
	supportsReasoningEffort: false,
	supportsUsageInStreaming: true,
	maxTokensField: "max_completion_tokens",
	requiresToolResultName: false,
	requiresAssistantAfterToolResult: false,
	requiresThinkingAsText: false,
	requiresReasoningContentOnAssistantMessages: false,
	thinkingFormat: "openai",
	openRouterRouting: {},
	vercelGatewayRouting: {},
	zaiToolStream: false,
	supportsStrictMode: true,
	sendSessionAffinityHeaders: false,
	supportsLongCacheRetention: false,
};

function buildGhostToolContext(api: "openai-completions" | "openai-responses"): Context {
	const assistant: AssistantMessage = {
		role: "assistant",
		api,
		provider: "openai",
		model: "test-model",
		usage,
		stopReason: "toolUse",
		timestamp: Date.now() - 2000,
		content: [
			{ type: "toolCall", id: "call_real", name: "echo", arguments: {} },
			{ type: "toolCall", id: "", name: "", arguments: { message: "stranded arguments" } },
		],
	};
	const validToolResult: ToolResultMessage = {
		role: "toolResult",
		toolCallId: "call_real",
		toolName: "echo",
		content: [{ type: "text", text: "ok" }],
		isError: false,
		timestamp: Date.now() - 1000,
	};
	const ghostToolResult: ToolResultMessage = {
		role: "toolResult",
		toolCallId: "",
		toolName: "",
		content: [{ type: "text", text: "Tool  not found" }],
		isError: true,
		timestamp: Date.now() - 900,
	};
	return {
		systemPrompt: "You are concise.",
		messages: [
			{ role: "user", content: "Use the tool.", timestamp: Date.now() - 3000 },
			assistant,
			validToolResult,
			ghostToolResult,
		],
	};
}

describe("OpenAI-compatible tool-call replay hardening", () => {
	it("drops orphan chat tool results created from empty-id ghost tool calls", () => {
		const payload = convertMessages(completionsModel, buildGhostToolContext("openai-completions"), completionsCompat);
		const toolMessages = payload.filter((message) => message.role === "tool");

		expect(toolMessages).toHaveLength(1);
		expect(toolMessages[0]).toMatchObject({ tool_call_id: "call_real", content: "ok" });
		expect(JSON.stringify(payload)).not.toContain("Tool  not found");
		expect(toolMessages.every((message) => message.tool_call_id.length > 0)).toBe(true);
	});

	it("drops orphan Responses tool outputs created from empty-id ghost tool calls", () => {
		const payload = convertResponsesMessages(
			responsesModel,
			buildGhostToolContext("openai-responses"),
			new Set(["openai"]),
		);
		const toolOutputs = payload.filter((item) => item.type === "function_call_output");

		expect(toolOutputs).toHaveLength(1);
		expect(toolOutputs[0]).toMatchObject({ call_id: "call_real", output: "ok" });
		expect(JSON.stringify(payload)).not.toContain("Tool  not found");
		expect(toolOutputs.every((item) => item.call_id.length > 0)).toBe(true);
	});

	it("normalizes empty pipe-prefixed Responses call ids consistently", () => {
		const assistant: AssistantMessage = {
			role: "assistant",
			api: "openai-responses",
			provider: "openai",
			model: "test-model",
			usage,
			stopReason: "toolUse",
			timestamp: Date.now() - 2000,
			content: [{ type: "toolCall", id: "|fc_bad", name: "echo", arguments: { message: "hello" } }],
		};
		const toolResult: ToolResultMessage = {
			role: "toolResult",
			toolCallId: "|fc_bad",
			toolName: "echo",
			content: [{ type: "text", text: "hello" }],
			isError: false,
			timestamp: Date.now() - 1000,
		};
		const payload = convertResponsesMessages(
			responsesModel,
			{ systemPrompt: "", messages: [assistant, toolResult] },
			new Set(["openai"]),
		);
		const callIds = payload
			.filter((item) => item.type === "function_call" || item.type === "function_call_output")
			.map((item) => item.call_id);

		expect(callIds).toHaveLength(2);
		expect(callIds.every((id) => id.length > 0)).toBe(true);
		expect(new Set(callIds).size).toBe(1);
	});

	it("normalizes empty pipe-prefixed chat tool ids consistently", () => {
		const assistant: AssistantMessage = {
			role: "assistant",
			api: "openai-completions",
			provider: "openai",
			model: "test-model",
			usage,
			stopReason: "toolUse",
			timestamp: Date.now() - 2000,
			content: [{ type: "toolCall", id: "|fc_bad", name: "echo", arguments: { message: "hello" } }],
		};
		const toolResult: ToolResultMessage = {
			role: "toolResult",
			toolCallId: "|fc_bad",
			toolName: "echo",
			content: [{ type: "text", text: "hello" }],
			isError: false,
			timestamp: Date.now() - 1000,
		};
		const payload = convertMessages(
			completionsModel,
			{ systemPrompt: "", messages: [assistant, toolResult] },
			completionsCompat,
		);
		const toolCallId = payload.find((message) => message.role === "assistant")?.tool_calls?.[0]?.id;
		const toolResultId = payload.find((message) => message.role === "tool")?.tool_call_id;

		expect(toolCallId).toBeDefined();
		expect(toolResultId).toBeDefined();
		expect(toolCallId).toBe(toolResultId);
		expect(toolCallId?.startsWith("|")).toBe(false);
	});
});
