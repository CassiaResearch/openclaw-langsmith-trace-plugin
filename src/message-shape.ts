/**
 * Translates OpenClaw's native message shape (from `@mariozechner/pi-ai`)
 * into the Anthropic/LangChain-canonical shape that LangSmith's
 * chat-model viewer renders natively.
 *
 * OpenClaw emits:
 *  - assistant content blocks like `{type: "toolCall", id, name, arguments}`
 *  - `{role: "toolResult", content: [...], toolCallId, toolName, isError}`
 *    messages at the top level
 *
 * LangSmith's viewer expects:
 *  - assistant blocks like `{type: "tool_use", id, name, input}`
 *  - `{role: "tool", tool_call_id, content}` messages
 *
 * Without this translation, tool calls and tool results render as
 * "unknown" blocks in the chat-message panel.
 */

type AnyRecord = Record<string, unknown>;

const TOOL_CALL_TYPES = new Set(["toolCall", "toolUse", "tool_use", "functionCall"]);

/** Normalize a single inbound OpenClaw message to the canonical wire shape. */
export function normalizeMessage(message: unknown): unknown {
  if (!isRecord(message)) return message;
  const role = message.role;

  if (role === "assistant") {
    return { ...message, content: normalizeAssistantContent(message.content) };
  }

  if (role === "toolResult") {
    const { toolCallId, toolName, isError, content, ...rest } = message as AnyRecord;
    const out: AnyRecord = {
      ...rest,
      role: "tool",
      content: normalizeToolResultContent(content),
    };
    if (typeof toolCallId === "string") out.tool_call_id = toolCallId;
    if (typeof toolName === "string") out.name = toolName;
    if (isError === true) out.status = "error";
    return out;
  }

  return message;
}

/**
 * Extract a LangChain-style `tool_calls` array from an assistant message —
 * LangSmith picks this up as a top-level hint even when content blocks are
 * in a non-canonical shape. Returns `undefined` when the message has no
 * tool calls.
 */
export function extractToolCalls(
  message: unknown,
): Array<{ id: string; name: string; args: unknown }> | undefined {
  if (!isRecord(message) || message.role !== "assistant") return undefined;
  const content = message.content;
  if (!Array.isArray(content)) return undefined;
  const calls: Array<{ id: string; name: string; args: unknown }> = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    const type = typeof block.type === "string" ? block.type : "";
    if (!TOOL_CALL_TYPES.has(type)) continue;
    const id = typeof block.id === "string" ? block.id : "";
    const name = typeof block.name === "string" ? block.name : "";
    if (!id || !name) continue;
    const args = "input" in block ? block.input : block.arguments;
    calls.push({ id, name, args });
  }
  return calls.length > 0 ? calls : undefined;
}

function normalizeAssistantContent(content: unknown): unknown {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content;
  return content.map((block) => {
    if (!isRecord(block)) return block;
    const type = typeof block.type === "string" ? block.type : "";
    if (TOOL_CALL_TYPES.has(type) && type !== "tool_use") {
      const { arguments: args, ...rest } = block as AnyRecord;
      return { ...rest, type: "tool_use", input: args };
    }
    return block;
  });
}

/**
 * LangSmith's viewer accepts tool-message content as either a string or an
 * array of typed content blocks. OpenClaw wraps tool results in an array of
 * text blocks; flatten to a string when every block is a simple text block
 * so the default renderer shows it without extra clicks.
 */
function normalizeToolResultContent(content: unknown): unknown {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content;
  const texts: string[] = [];
  for (const block of content) {
    if (!isRecord(block)) return content;
    if (block.type !== "text") return content;
    if (typeof block.text !== "string") return content;
    texts.push(block.text);
  }
  return texts.join("");
}

function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
