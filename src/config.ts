export interface PluginConfig {
  apiKey: string | undefined;
  endpoint: string;
  projectName: string;
  traceAgentTurns: boolean;
  traceToolCalls: boolean;
  samplingRate: number;
  failedTracesDir: string | undefined;
  debug: boolean;
}

const DEFAULT_ENDPOINT = "https://api.smith.langchain.com";
const DEFAULT_PROJECT = "openclaw";
const LANGSMITH_API_KEY_ENV = "LANGSMITH_API_KEY";

export function readPluginConfig(raw: Record<string, unknown> | undefined): PluginConfig {
  const src = raw ?? {};
  const apiKey = readString(src.langsmithApiKey) ?? process.env[LANGSMITH_API_KEY_ENV];

  return {
    apiKey: apiKey && apiKey.length > 0 ? apiKey : undefined,
    endpoint: readString(src.langsmithEndpoint) ?? DEFAULT_ENDPOINT,
    projectName: readString(src.projectName) ?? DEFAULT_PROJECT,
    traceAgentTurns: readBool(src.traceAgentTurns, true),
    traceToolCalls: readBool(src.traceToolCalls, true),
    samplingRate: readSamplingRate(src.tracingSamplingRate),
    failedTracesDir: readString(src.failedTracesDir),
    debug: readBool(src.debug, false),
  };
}

function readString(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

function readBool(v: unknown, fallback: boolean): boolean {
  return typeof v === "boolean" ? v : fallback;
}

function readSamplingRate(v: unknown): number {
  if (typeof v !== "number" || Number.isNaN(v)) return 1;
  return Math.min(1, Math.max(0, v));
}
