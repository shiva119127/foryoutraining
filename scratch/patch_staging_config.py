from pathlib import Path

# 1. Update environment.ts
env_path = Path("/Users/shivamsingh/Desktop/foryou/src/config/environment.ts")
env_lines = env_path.read_text(encoding="utf-8").splitlines()

# Replace AiEnvironment definition (line ~8)
for idx, line in enumerate(env_lines):
    if line.startswith("export type AiEnvironment"):
        env_lines[idx] = "export type AiEnvironment = 'production' | 'staging' | 'development' | 'test';"
        break

# Replace primaryProvider type (line ~20)
for idx, line in enumerate(env_lines):
    if "primaryProvider: 'gemini' | 'openai';" in line:
        env_lines[idx] = "    primaryProvider: 'gemini' | 'openai' | 'phase6d-adapter';\n    stagingEndpoint?: string;\n    activeStagingAdapter?: string;"
        break

# Replace AI Environment resolution (around line 80-90)
start_res = -1
end_res = -1
for idx, line in enumerate(env_lines):
    if line.strip() == "// AI Environment resolution":
        start_res = idx
    if start_res != -1 and line.strip() == "}":
        end_res = idx
        break

if start_res != -1 and end_res != -1:
    new_res_block = """  // AI Environment resolution
  const rawAiEnv = process.env.AI_ENV?.toLowerCase() || process.env.NODE_ENV;
  let aiEnv: AiEnvironment = 'development';
  if (rawAiEnv === 'production') {
    aiEnv = 'production';
  } else if (rawAiEnv === 'staging') {
    aiEnv = 'staging';
  } else if (rawAiEnv === 'test') {
    aiEnv = 'test';
  } else {
    aiEnv = 'development';
  }

  // CRITICAL PRODUCTION SAFETY GATE:
  // Production MUST NEVER use local adapter!
  if (aiEnv === 'production') {
    if (process.env.STAGING_PHASE6D === 'true' || process.env.USE_PHASE6D_ADAPTER === 'true' || process.env.AI_PROVIDER === 'phase6d') {
      throw new Error(
        '[CRITICAL SAFETY GATE] Phase 6D local adapter cannot be activated in AI_ENV=production. ' +
        'Production strictly requires validated cloud providers (Gemini / OpenAI).'
      );
    }
  }"""
    env_lines = env_lines[:start_res] + new_res_block.splitlines() + env_lines[end_res+1:]

# Replace llm config in cachedConfig
for idx, line in enumerate(env_lines):
    if "primaryProvider: hasGemini ? 'gemini' : 'openai'," in line:
        env_lines[idx] = """      primaryProvider: (aiEnv === 'staging' || process.env.STAGING_PHASE6D === 'true')
        ? 'phase6d-adapter'
        : (hasGemini ? 'gemini' : 'openai'),
      activeGeminiModel: process.env.ACTIVE_GEMINI_MODEL || 'gemini-3.5-flash-lite',
      activeOpenAiModel: process.env.ACTIVE_OPENAI_MODEL || 'gpt-4o-mini',
      stagingEndpoint: process.env.PHASE6D_ENDPOINT || 'http://127.0.0.1:8765',
      activeStagingAdapter: process.env.STAGING_ADAPTER_PATH || 'outputs/flowersforyou-adapter-10k'"""
        # Remove old activeGeminiModel and activeOpenAiModel lines right after
        if "activeGeminiModel:" in env_lines[idx+1]:
            del env_lines[idx+1]
        if "activeOpenAiModel:" in env_lines[idx+1]:
            del env_lines[idx+1]
        break

env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
print("environment.ts updated successfully.")

# 2. Update llmProvider.ts
prov_path = Path("/Users/shivamsingh/Desktop/foryou/src/ai/provider/llmProvider.ts")
prov_lines = prov_path.read_text(encoding="utf-8").splitlines()

# Update LlmClient providerName type
for idx, line in enumerate(prov_lines):
    if "readonly providerName: 'gemini' | 'openai' | 'deterministic-fallback';" in line:
        prov_lines[idx] = "  readonly providerName: 'gemini' | 'openai' | 'deterministic-fallback' | 'phase6d-adapter';"
        break

# Add Phase6DStagingClient class before ResilientLlmOrchestrator
resilient_idx = -1
for idx, line in enumerate(prov_lines):
    if "export class ResilientLlmOrchestrator" in line:
        resilient_idx = idx
        break

phase6d_class = """/**
 * Phase 6D Staging Adapter Client
 * Executes inference against the local Phase 6D PEFT adapter endpoint.
 * Bounded by circuit breaker and local timeout safeguards.
 */
export class Phase6DStagingClient implements LlmClient {
  readonly providerName = 'phase6d-adapter' as const;
  readonly modelName = 'qwen2.5-0.5b-phase6d';
  private endpoint: string;

  constructor(endpoint: string = 'http://127.0.0.1:8765') {
    this.endpoint = endpoint;
  }

  async generateText(
    prompt: string,
    promptVersion: string,
    operationName: string,
    requestId?: string
  ): Promise<{ text: string; metadata: ExecutionMetadata }> {
    const startTime = Date.now();
    const reqId = requestId || `req_stg_${Date.now()}`;
    const breaker = providerCircuitBreakers['phase6d-adapter'] || providerCircuitBreakers['gemini'];

    if (breaker && !breaker.canExecute()) {
      throw new AiProviderError('Local Phase 6D adapter circuit breaker is OPEN.', {
        operation: operationName,
        provider: 'phase6d-adapter',
        model: this.modelName,
        prompt_version: promptVersion,
        latency_ms: 0,
        success: false,
        failure_reason: 'Circuit breaker fast-fail',
        request_id: reqId
      });
    }

    try {
      const result = await withExponentialBackoff(
        async () => {
          const res = await fetch(`${this.endpoint}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt,
              messages: [
                { role: 'system', content: 'You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details.' },
                { role: 'user', content: prompt }
              ],
              temperature: 0.3,
              max_tokens: 150
            }),
            signal: AbortSignal.timeout(8000)
          });

          if (!res.ok) {
            throw new Error(`Phase 6D adapter HTTP ${res.status}`);
          }

          const json = await res.json();
          const text = json.text || '';
          const latency = Date.now() - startTime;
          if (breaker) breaker.recordSuccess(latency);

          return {
            text,
            metadata: {
              operation: operationName,
              provider: 'phase6d-adapter' as const,
              model: this.modelName,
              prompt_version: promptVersion,
              latency_ms: latency,
              success: true,
              token_count: json.token_count || 0,
              cost_usd: 0,
              request_id: reqId
            }
          };
        },
        {
          maxRetries: 1,
          initialDelayMs: 400,
          operationName: `Phase6D-${operationName}`
        }
      );

      return result;
    } catch (err: any) {
      const latency = Date.now() - startTime;
      if (breaker) breaker.recordFailure(err, latency);
      throw new AiProviderError(`Phase 6D adapter execution failed: ${err.message}`, {
        operation: operationName,
        provider: 'phase6d-adapter',
        model: this.modelName,
        prompt_version: promptVersion,
        latency_ms: latency,
        success: false,
        failure_reason: err.message,
        request_id: reqId
      });
    }
  }

  async generateStructured<T>(
    prompt: string,
    promptVersion: string,
    validator: (parsedJson: any) => T,
    operationName: string,
    requestId?: string
  ): Promise<LlmExecutionResult<T>> {
    const { text, metadata } = await this.generateText(prompt, promptVersion, operationName, requestId);
    const cleaned = extractJsonString(text);
    try {
      const parsed = JSON.parse(cleaned);
      const data = validator(parsed);
      return {
        data,
        rawText: text,
        metadata
      };
    } catch (err: any) {
      throw new AiProviderError(`JSON validation error from Phase 6D adapter: ${err.message}`, {
        ...metadata,
        success: false,
        failure_reason: `JSON schema failure: ${err.message}`
      });
    }
  }
}
"""

if resilient_idx != -1:
    prov_lines = prov_lines[:resilient_idx] + phase6d_class.splitlines() + prov_lines[resilient_idx:]

# Update getActiveLlmClient()
get_active_idx = -1
for idx, line in enumerate(prov_lines):
    if "export function getActiveLlmClient(): LlmClient | null {" in line:
        get_active_idx = idx
        break

if get_active_idx != -1:
    staging_selection = """  // CRITICAL PRODUCTION SAFETY GATE:
  // Production MUST NEVER use local adapter!
  if (config.aiEnv === 'production') {
    if (process.env.STAGING_PHASE6D === 'true' || process.env.USE_PHASE6D_ADAPTER === 'true' || process.env.AI_PROVIDER === 'phase6d') {
      throw new Error(
        '[CRITICAL SAFETY GATE] Phase 6D local adapter cannot be activated in AI_ENV=production. ' +
        'Production strictly requires validated cloud providers (Gemini / OpenAI).'
      );
    }
  }

  // Staging Mode configuration: local Phase 6D adapter
  if (config.aiEnv === 'staging' || (process.env.STAGING_PHASE6D === 'true' && config.aiEnv !== 'production')) {
    const endpoint = config.llm.stagingEndpoint || process.env.PHASE6D_ENDPOINT || 'http://127.0.0.1:8765';
    return new Phase6DStagingClient(endpoint);
  }"""
    prov_lines.insert(get_active_idx + 1, staging_selection)

prov_path.write_text("\n".join(prov_lines) + "\n", encoding="utf-8")
print("llmProvider.ts updated successfully.")
