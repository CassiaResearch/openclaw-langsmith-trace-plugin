# openclaw-langsmith-trace-plugin

LangSmith tracing plugin for [OpenClaw](https://github.com/openclaw/openclaw), built on the official [`langsmith`](https://www.npmjs.com/package/langsmith) SDK. Every agent turn, inner LLM call, tool invocation, and subagent is traced as a `RunTree` so you can see the full conversation shape in LangSmith's Threads view.

## What gets traced

For every agent turn this plugin produces a `chain` root run named `agent_turn`, with one `llm` child per inner LLM call and `tool` children nested under the LLM that invoked them:

```
agent_turn                    (chain, root)
├── ChatAnthropic             (llm)
│   ├── <tool_name>           (tool)
│   └── <tool_name>           (tool)
├── ChatAnthropic             (llm)
│   └── <tool_name>           (tool)
└── subagent:<key>            (chain, subagent turns)
```

- LLM runs use LangChain's chat-class names (`ChatAnthropic`, `ChatOpenAI`, `ChatGoogleGenerativeAI`, …) so LangSmith's UI renders them with the standard chat-model card (model icon, token panel, message viewer) — the same way a LangGraph trace looks.
- Root is always `chain` type — tokens live on LLM children only, so the cost calculator doesn't double-count.
- Every run carries `metadata.thread_id = sessionKey`, so turns in the same session group under LangSmith's Threads view.
- Token usage is emitted in canonical `usage_metadata` form (`cache_read`, `cache_creation`, `reasoning` breakdowns) so the cost calculator works for Anthropic prompt caching and OpenAI reasoning models.

### Known gap: tool catalog

Tool **invocations** appear as child runs under the LLM that called them (with params and result). The tool **catalog** — i.e. the list of tool names/descriptions/schemas the model had access to on each call — is **not** currently included in the trace. `PluginHookLlmInputEvent` doesn't expose the resolved tool list, so the plugin has no way to read it. Once OpenClaw adds `tools` to that event, the plugin will attach it as `metadata.available_tools` on the root and `inputs.tools` on each inner LLM child (matching how LangGraph renders it). There's a `TODO` in `src/turn-recorder.ts:onTurnStart` marking the exact insertion point.

## Install

Install via `openclaw plugins install` — any of the standard target forms work. See the [OpenClaw plugins CLI docs](https://github.com/openclaw/openclaw/blob/main/docs/cli/plugins.md) for the full reference (ClawHub, npm, local path, marketplace, `--pin`, `--force`, etc.).

```bash
# From a local clone (most common while iterating):
openclaw plugins install ./path/to/openclaw-langsmith-trace-plugin

# Or from npm / ClawHub once published:
openclaw plugins install openclaw-langsmith-trace

openclaw gateway restart
```

Set your API key:

```bash
openclaw config set plugins.entries.openclaw-langsmith-trace.config.langsmithApiKey "lsv2_pt_..."
```

Or export `LANGSMITH_API_KEY` in the gateway's environment.

## Configuration

| Option                | Type    | Default                           | Description                                                                   |
| --------------------- | ------- | --------------------------------- | ----------------------------------------------------------------------------- |
| `langsmithApiKey`     | string  | _unset_                           | Required to enable tracing. Falls back to `$LANGSMITH_API_KEY` when unset.    |
| `langsmithEndpoint`   | string  | `https://api.smith.langchain.com` | Override for self-hosted LangSmith.                                           |
| `projectName`         | string  | `openclaw`                        | Project that receives traces.                                                 |
| `traceAgentTurns`     | boolean | `true`                            | Trace turn + inner LLM call tree.                                             |
| `traceToolCalls`      | boolean | `true`                            | Trace tool calls as children of the invoking LLM.                             |
| `tracingSamplingRate` | number  | `1.0`                             | Fraction of turns to trace (0.0–1.0).                                         |
| `failedTracesDir`     | string  | _SDK default_                     | Directory for SDK's failed-batch spool (used by `langsmith replay-failures`). |
| `debug`               | boolean | `false`                           | Verbose `[langsmith]` logs.                                                   |

## How it works

This plugin is a thin adapter between the OpenClaw hook API and the LangSmith SDK:

- `session_end` — closes any turn still open for that session.
- `llm_input` — opens a `chain` root `RunTree` for the turn. If a turn already exists for that session (compaction retry), the previous root is closed with `"Compacted and retried"` before a fresh one opens.
- `before_message_write` — every assistant message spawns a new `llm` child `RunTree`. The message buffer passed to each child is the exact context the model saw on that call.
- `before_tool_call` / `after_tool_call` — tool runs attach as children of the most recent LLM run (the one that invoked them), not of the root.
- `subagent_spawned` / `subagent_ended` — `subagent_spawned` captures the start timestamp; `subagent_ended` attaches a `chain` child to the parent root with the real duration.
- `agent_end` — closes the root with aggregated token totals.

The SDK handles all ID / trace_id / dotted_order plumbing, batching (250 ms aggregation by default), retries, and failed-batch disk persistence.

## Development

```bash
npm install
npm run typecheck
npm run test
```

The plugin entry is `./src/index.ts` — OpenClaw loads the TypeScript source directly via its plugin loader, so there is no build step.

## License

MIT
