# FlowersForYou Phase 6D — Inference Infrastructure and Shadow-Test Plan

**Document:** Inference Infrastructure & Shadow-Test Preparation Proposal  
**Date:** October 16, 2026  
**Status:** Proposal & Plan (Review-Only — No Production Deployment)  
**Target Model:** `Qwen/Qwen2.5-0.5B-Instruct` + Phase 6D LoRA Adapter (`outputs/flowersforyou-adapter-10k`)  
**Host Application:** `~/Desktop/foryou`  
**Primary Source of Truth:** `reports/phase6d_deployment_readiness_audit.md`

---

## Executive Summary

The Phase 6D deployment-readiness audit certified the adapter as **STAGING-VALIDATED**, but identified a critical operational bottleneck: the single-worker staging inference daemon serializes requests under a global Python lock (`gpu_lock`), making it incapable of handling concurrent multi-tenant traffic.

This document establishes the end-to-end infrastructure, serving architecture, read-only shadow evaluation pipeline, and load-testing methodology required before any limited production canary can be considered.

### Key Guarantees
- **Zero Production Risk:** The cloud model (`gemini-3.5-flash-lite` with fallback to `gpt-4o-mini`) remains the active production default.
- **Fail-Fast Enforcement:** Double-layer compile-time and runtime safety gates in `src/config/environment.ts` and `src/ai/provider/llmProvider.ts` prevent accidental local-adapter activation in production.
- **Zero User Memory Mutation in Shadow Mode:** The shadow evaluation pipeline executes asynchronously in read-only trial mode. Phase 6D extraction results are recorded strictly to comparison logs and **never** invoke `MemoryManager.reconcileAndSave()`, ensuring zero database mutations or side effects.
- **Actionable Runnable Test Suite:** Complete k6 (`scratch/load_test_phase6d.js`) and Locust (`scratch/locustfile.py`) test plans simulating 50 concurrent virtual users across a 15-minute profile (warm-up, steady state, burst, and cooldown).

---

## 1. Current Architecture & Deployment Environment Audit

### 1.1 Existing Staging Inference Server
- **File:** `scratch/staging_inference_server.py`
- **Architecture:** Python `ThreadingHTTPServer` on port `8765`.
- **Concurrency Model:** Uses `gpu_lock = threading.Lock()` around PyTorch `model.generate()`.
- **Critical Limitation:** On a single worker, concurrent requests queue sequentially. At an average generation latency of 3.2s per request, a burst of 6 concurrent users causes the 6th request to hit the client's 18s abort timeout (`AbortSignal.timeout(18000)`).
- **Cold-Start Latency:** 5,612 ms on Apple Silicon MPS backend.

### 1.2 Application Provider Routing
- **File:** `src/ai/provider/llmProvider.ts` (`getActiveLlmClient()`)
- **Routing Rules:**
  1. `AI_ENV === 'production'`: Throws `[CRITICAL SAFETY GATE]` if staging flags are present; otherwise returns `ResilientLlmOrchestrator(primary, secondary)`.
  2. `AI_ENV === 'staging'`: Returns `Phase6DStagingClient(http://127.0.0.1:8765)`.
  3. `AI_ENV === 'test'`: Returns `null` (routes directly to local deterministic parser).
- **Failure Handling:** `Phase6DStagingClient` wraps HTTP calls with `withExponentialBackoff` (1 retry, 400ms delay) and circuit-breaker telemetry. If the daemon is unreachable, the application catches `AiProviderError` and falls back cleanly to deterministic local parsing.

### 1.3 Deployment Environment
- **Development/Staging Hardware:** macOS (Apple Silicon arm64, Metal Performance Shaders / MPS, FP16).
- **Target Production Hardware:** Linux x86_64 or Linux arm64 (Cloud VM / Container runtime).
  - Production DB: Supabase Cloud (PostgreSQL 15).
  - Application Hosting: Next.js on Node.js v26.

---

## 2. Inference Server Compatibility Assessment

Rather than assuming generic tools like vLLM or Triton will work everywhere, candidate inference servers were evaluated against the specific model (`Qwen2.5-0.5B-Instruct`), adapter format (LoRA 35.2 MB safetensors), and target hardware.

### 2.1 Model & Precision Constraints
- **Model Size:** 490 million parameters. In FP16/BF16, total model size is ~0.98 GB.
- **LoRA Adapter:** $r=16, \alpha=32$, targeting 7 projection layers (`q, k, v, o, gate, up, down`). Total LoRA weights: ~8.8M parameters (35.2 MB safetensors).
- **VRAM / RAM Footprint:** At 1 GB total footprint, quantization (INT4/AWQ) yields negligible memory benefit while risking JSON syntax degradation. **FP16 or BF16 is the recommended operational precision.**

### 2.2 Inference Server Evaluation Matrix

| Inference Engine | Target Hardware Compatibility | LoRA Compatibility | Concurrency & Dynamic Batching | Resource Footprint | Suitability Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vLLM** (`vllm.entrypoints.openai.api_server`) | • Linux x86_64 + NVIDIA GPU (CUDA 12.x): **100% Native**<br>• Linux CPU: Experimental<br>• macOS MPS: **Incompatible** (No production MPS kernel support) | Native dynamic LoRA via `--enable-lora` and `--lora-modules phase6d=...` | **Best-in-class continuous batching** (PagedAttention); handles 100+ concurrent streams on a single GPU. | High (Container image is ~12 GB due to CUDA/PyTorch) | **PRIMARY RECOMMENDATION for Linux GPU Cloud Canary** |
| **llama.cpp** (`llama-server`) | • Linux x86_64 / arm64 (CPU & CUDA): **100% Native**<br>• macOS (Metal/MPS): **100% Native**<br>• Windows: Supported | Supported via pre-merged weights (`merge_and_unload`) exported to GGUF format. | **Native continuous batching** (`--cont-batching`) and multi-slot processing (`-np 8`). | **Ultra-lightweight** (<50 MB binary, <1.2 GB RAM total) | **PRIMARY RECOMMENDATION for Cost-Optimized Cloud CPU & Local macOS Staging** |
| **Hugging Face TGI** | • Linux x86_64 + NVIDIA GPU: **Native**<br>• macOS MPS: **Incompatible** | Supported via LoRA adapter mount or merged weights. | Continuous batching with FlashAttention-2. | Medium-High (~8 GB container) | Viable alternative to vLLM on Linux GPU. |
| **Triton Inference Server** | • Linux x86_64 + NVIDIA GPU: **Native**<br>• macOS MPS: **Incompatible** | Complex setup via Python/vLLM backend. | Dynamic batch scheduler, gRPC and HTTP. | Heavy (>15 GB container) | **Over-engineered** for a single 0.5B model. Not recommended. |
| **Custom Async FastAPI + PyTorch Worker Pool** | • Cross-platform (Linux, macOS MPS, Windows): **100% Compatible** | Native `PeftModel.from_pretrained` (current setup). | Bounded thread/process worker pool (e.g. 4-8 parallel workers). Lacks continuous token batching. | Lightweight (~2 GB RAM) | Ideal bridge for local macOS development and testing. |

### 2.3 Concrete Deployment Recommendations
1. **For Production Cloud Canary (GPU Instance — e.g., AWS `g5.xlarge` with NVIDIA A10G or `g6.large` with L4):**
   - Deploy **vLLM** in Docker with `--enable-lora`. Provides continuous batching, sub-50ms token latency, and native Prometheus metrics.
2. **For Cost-Optimized Production Cloud Canary (CPU Instance — e.g., AWS `c6i.xlarge` 4 vCPU / 8 GB RAM):**
   - Deploy **llama.cpp (`llama-server`)** running the merged Phase 6D GGUF (`Qwen2.5-0.5B-Phase6D-FP16.gguf`). Provides 8 parallel slots with continuous batching for under $30/month.
3. **For Local Staging & Development Parity (macOS Apple Silicon):**
   - Run the custom asynchronous FastAPI server with bounded queues (`scratch/production_inference_server.py`) or `llama-server` compiled with Metal acceleration.

---

## 3. Containerized Serving Architecture

### 3.1 Architecture Overview
The containerized inference service encapsulates model serving, health probing, queue admission control, request timeouts, and graceful shutdown into an isolated microservice.

```
                           +-------------------------------------------------------+
                           |           Phase 6D Containerized Service              |
                           |                                                       |
[Client HTTP Request] ---> | [FastAPI / Nginx Ingress]                             |
                           |   |                                                   |
                           |   +--> [Admission Control & Queue Depth Gate]         |
                           |   |      • If active + queue >= 72: Return HTTP 503   |
                           |   |                                                   |
                           |   +--> [Concurrency Semaphore: max 8 parallel]        |
                           |   |      • Client disconnect check (HTTP 499)         |
                           |   |      • Per-request deadline (10s Timeout)         |
                           |   |                                                   |
                           |   +--> [Engine: vLLM or PyTorch CausalLM]             |
                           |   |      • FP16 / BF16 Precision                      |
                           |   |      • Base: Qwen2.5-0.5B + LoRA Checkpoint 625   |
                           |   |                                                   |
                           |   +--> [/health/live]  [/health/ready]  [/metrics]    |
                           +-------------------------------------------------------+
```

### 3.2 Dockerfile Specification
Reference container build for CPU/GPU deployment (`Dockerfile.inference`):

```dockerfile
# Multi-stage production container build
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --no-cache-dir --user -r requirements-inference.txt

FROM python:3.10-slim-bookworm AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create unprivileged service user
RUN groupadd -r aiuser && useradd -r -g aiuser -u 10001 aiuser

COPY --from=builder /root/.local /home/aiuser/.local
ENV PATH=/home/aiuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy service code and weights
COPY scratch/production_inference_server.py /app/server.py
COPY outputs/flowersforyou-adapter-10k /app/adapter

RUN chown -R aiuser:aiuser /app
USER aiuser

EXPOSE 8000

# Health check probes
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health/ready || exit 1

ENTRYPOINT ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 3.3 Docker Compose Specification
Reference compose configuration with strict resource limits (`docker-compose.inference.yml`):

```yaml
version: '3.8'

services:
  phase6d-inference:
    build:
      context: .
      dockerfile: Dockerfile.inference
    image: flowersforyou/phase6d-inference:v1.0.0
    container_name: phase6d-inference-service
    ports:
      - "8000:8000"
    environment:
      - MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
      - ADAPTER_PATH=/app/adapter
      - MAX_CONCURRENT_REQUESTS=8
      - MAX_QUEUE_CAPACITY=64
      - REQUEST_TIMEOUT_SECONDS=10.0
      - DRAIN_TIMEOUT_SECONDS=15.0
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4096M
        reservations:
          cpus: '2.0'
          memory: 2048M
    restart: on-failure:5
    stop_grace_period: 20s
```

### 3.4 Operational Safeguards
1. **Health Probes:**
   - `/health/live`: Returns `200 OK` if the process is running. Used by Kubernetes/Docker for process restart decisions.
   - `/health/ready`: Returns `200 OK` only after the base model and LoRA adapter are loaded into memory and ready for inference. Returns `503 Service Unavailable` during startup or graceful draining.
2. **Bounded Request Queue & Backpressure:**
   - Maximum 8 concurrent model executions (`asyncio.Semaphore(8)`).
   - Maximum queue buffer of 64 requests.
   - If total in-flight requests exceed 72 ($8 + 64$), incoming requests are rejected immediately with **HTTP 503 Service Unavailable** (`detail: "Inference request queue saturated. Please retry with exponential backoff."`). This prevents memory exhaustion and unmanageable latency cascades.
3. **Timeouts & Client Disconnect Handling:**
   - Requests waiting in the queue check `await request.is_disconnected()`. If the user has closed the browser tab or canceled the request, execution is abandoned before GPU cycles are wasted.
   - Model generation is wrapped with `asyncio.wait_for(timeout=10.0)`. Inferences exceeding 10 seconds throw an HTTP 504 Gateway Timeout.
4. **Graceful Shutdown:**
   - Handles `SIGTERM` and `SIGINT`.
   - Flips readiness probe to `503 shutting_down` to stop ingress traffic.
   - Grants an explicit 15-second drain window to allow active inferences to complete cleanly.

---

## 4. Read-Only Shadow Mode Architecture for Memory Extraction

### 4.1 Zero-Side-Effect Guarantee
The shadow mode allows Phase 6D to be evaluated against real-world user utterances in parallel with the primary cloud provider without risking customer data or experience.

```
[User Utterance via API]
          |
          v
[MemoryExtractor.extract()]
          |
          +--------------------------------------------------------+
          |                                                        |
          v (Synchronous Primary Path)                             v (Non-Blocking Fire-and-Forget)
[Primary Cloud Provider: Gemini 3.5]                      [ShadowMemoryExtractor.triggerShadowEvaluation()]
          |                                                        |
          v (Latency: ~1.0s)                                       v (setImmediate background task)
[Validated Structured Memory Records]                     [Phase 6D Staging Endpoint: /generate]
          |                                                        |
          v                                                        v (Latency: ~2.8s)
[MemoryManager.reconcileAndSave()]                        [Sandbox SchemaValidator.validateMemoryExtraction()]
          |                                                        |
          v                                                        v
[Live User Database: Preferences Updated]                 [Differential Analysis: Cloud vs Phase 6D]
          |                                                        |
          v                                                        v
[Response Returned to User: IMMEDIATE]                    [Append to shadow_extraction_logs.jsonl]
                                                                   • ZERO User Memory Mutation
                                                                   • ZERO Live Side Effects
                                                                   • All Errors Caught & Suppressed
```

### 4.2 Architectural Rules of Shadow Mode
1. **Fire-and-Forget Execution:**
   - Dispatched via `setImmediate()` or a detached Promise in Node.js.
   - The user's HTTP response is returned immediately following the primary cloud provider's completion. Shadow latency **never** adds to user-perceived response time.
2. **Strict Memory Isolation:**
   - Phase 6D extraction records are run through `SchemaValidator.validateMemoryExtraction` purely for analysis.
   - **`MemoryManager.reconcileAndSave()` is NEVER called on shadow outputs.**
   - Shadow records are never written to `user_preferences`, `user_interests`, or `raw_records`.
3. **Error & Timeout Containment:**
   - If the shadow daemon crashes, returns HTTP 500, or times out (8.0s deadline), the error is recorded to telemetry and silently suppressed. The user's conversation turn is unaffected.
4. **Anonymized Audit Log:**
   - Records are written to `reports/shadow_logs/shadow_eval_YYYY-MM-DD.jsonl`.
   - `userId` is anonymized via SHA-256 (`crypto.createHash('sha256').update(userId).digest('hex').slice(0, 16)`).
   - Log contains: prompt length, primary extracted dimensions, shadow extracted dimensions, match status (`exact_match`, `partial_match`, `disagreement`, `shadow_failure`), and explicit diff strings.

---

## 5. 50-User / 15-Minute Load Test Plan

### 5.1 Test Objectives
1. Profile throughput and latency distributions ($p_{50}, p_{95}, p_{99}$) under realistic concurrent load (50 simulated users).
2. Validate queue admission control, backpressure, and graceful 503 rejection during a burst spike (75 users).
3. Verify server stability, absence of memory leaks, and recovery following load spikes.

### 5.2 Test Schedule (15 Minutes)

```
Users (VUs)
  80 |                             [Burst Spike: 75 VUs]
  60 |                 +-----------------------------+
  40 |         +-------+ [Steady State: 50 VUs]      +-------+
  20 | +-------+                                             +-------+
   0 +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---> Time (mins)
       0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
```

| Phase | Duration | Virtual Users (VUs) | Target Behavior |
| :--- | :---: | :---: | :--- |
| **1. Warm-Up** | 2 min | 0 $\rightarrow$ 10 | Model warmup, JIT compilation, buffer allocation |
| **2. Ramp-Up** | 3 min | 10 $\rightarrow$ 50 | Increasing concurrency, queue depth building |
| **3. Steady State** | 6 min | 50 (sustained) | Sustained operational throughput, KV-cache stress |
| **4. Peak Burst Stress**| 1 min | 50 $\rightarrow$ 75 | Stress test queue admission and 503 backpressure |
| **5. Burst Recovery** | 1 min | 75 $\rightarrow$ 50 | Verify queue drains cleanly without crashing |
| **6. Cooldown** | 2 min | 50 $\rightarrow$ 0 | Drain in-flight requests, verify memory reclamation |

### 5.3 Test Utterance Distribution (10 Realistic Archetypes)
The load tests cycle through 10 canonical input templates reflecting real user interactions:
1. `Single Interest`: *"I enjoy pottery workshops and quiet nature walks."*
2. `Multi-Interest + Area`: *"Looking for specialty coffee tasting and artisanal sourdough baking around Indiranagar."*
3. `Numeric Budget INR`: *"Update my spending limit: my maximum budget is ₹1500 per person in Bandra West for indie board games."*
4. `Neighborhood & Budget Correction`: *"Actually, scratch Indiranagar—I moved to Koramangala now, and please raise my maximum budget to ₹2000."*
5. `Explicit Clearing`: *"Please clear my budget limit, I don't want any budget ceiling."*
6. `Disclaimed Preference`: *"I really enjoy modern strategy board games, but I haven't decided on any location or budget yet."*
7. `Dietary Boundary`: *"I am strictly vegan and need dairy-free chocolate tasting workshops."*
8. `Accessibility Boundary`: *"Require wheelchair accessibility for all outings around Church Street."*
9. `Compound Outing Intent`: *"Outdoor cycling around Cubbon Park this Saturday morning, keeping it under ₹500."*
10. `Additive Interest`: *"Also add linocut printmaking to my interests, keep my area as Koramangala."*

### 5.4 Test Scripts Created
- **k6 Test Script:** [`scratch/load_test_phase6d.js`](file:///Users/shivamsingh/Downloads/flowersforyou-finetuning/scratch/load_test_phase6d.js)
  - Configured with custom metrics: `phase6d_extraction_duration_ms`, `phase6d_error_rate`, `phase6d_valid_json_rate`.
  - Configured with strict thresholds: $p_{50} < 3,000\text{ ms}, p_{95} < 5,000\text{ ms}, p_{99} < 7,500\text{ ms}$, failure rate $< 0.5\%$.
- **Locust Test Script:** [`scratch/locustfile.py`](file:///Users/shivamsingh/Downloads/flowersforyou-finetuning/scratch/locustfile.py)
  - Configured with weighted tasks (60% standard extraction, 20% corrections/clearings, 20% safety boundaries).
  - Simulates 1.0s to 3.0s realistic user think time between conversation turns.

### 5.5 Runnable Test Execution Commands
```bash
# -------------------------------------------------------------
# Option A: Run load test using k6 (Recommended)
# -------------------------------------------------------------
# Install k6 (macOS): brew install k6
# Run 15-minute load test against staging inference server:
k6 run scratch/load_test_phase6d.js --env TARGET_URL=http://127.0.0.1:8765/generate

# -------------------------------------------------------------
# Option B: Run load test using Locust
# -------------------------------------------------------------
# Install locust: pip install locust
# Run headless 15-minute test with 50 users (spawn rate 5 users/sec):
locust -f scratch/locustfile.py --headless \
  --users 50 \
  --spawn-rate 5 \
  --run-time 15m \
  --host http://127.0.0.1:8765 \
  --csv reports/load_test_locust_results
```

---

## 6. Observability, Metrics & Alerting Framework

### 6.1 Core Metric Taxonomy

| Metric Category | Metric Name | Type | Description | Target SLA / Alert Threshold |
| :--- | :--- | :---: | :--- | :--- |
| **Latency ($p_{50}$)** | `phase6d_latency_p50_ms` | Gauge | Median end-to-end extraction latency | Target: $< 2,500\text{ ms}$; Alert: $> 3,500\text{ ms}$ |
| **Latency ($p_{95}$)** | `phase6d_latency_p95_ms` | Gauge | 95th percentile extraction latency | Target: $< 4,500\text{ ms}$; Alert: $> 6,000\text{ ms}$ |
| **Latency ($p_{99}$)** | `phase6d_latency_p99_ms` | Gauge | 99th percentile tail latency | Target: $< 7,000\text{ ms}$; Alert: $> 8,500\text{ ms}$ |
| **Throughput** | `phase6d_requests_per_sec` | Rate | Number of completed extractions per second | Min: $> 10\text{ req/s}$ on GPU |
| **Token Velocity** | `phase6d_tokens_per_sec` | Rate | Generation tokens produced per second | $> 250\text{ tokens/s}$ across batch |
| **Queue Depth** | `phase6d_queue_depth` | Gauge | Current requests waiting in admission queue | Normal: $< 16$; Warning: $> 32$; Critical: $> 56$ |
| **Queue Saturation** | `phase6d_queue_rejections_total` | Counter | Number of HTTP 503 backpressure rejections | Target: $0$ during steady state; Alert: $> 1\%$ |
| **Timeout Rate** | `phase6d_timeouts_total` | Counter | Number of requests hitting 10s deadline (504) | Target: $0$; Alert: $> 0.1\%$ |
| **Malformed JSON** | `phase6d_json_parse_errors_total` | Counter | Model output lacking valid parseable JSON block | Target: $< 0.5\%$; Alert: $> 2.0\%$ |
| **Validation Failures**| `phase6d_schema_failures_total` | Counter | JSON parses but fails `SchemaValidator` typing | Target: $< 0.5\%$; Alert: $> 2.0\%$ |

### 6.2 Cloud Baseline Disagreement Metrics (Shadow Mode)

| Shadow Metric | Calculation Formula | Target Threshold | Interpretation / Alert |
| :--- | :--- | :---: | :--- |
| **Exact Match Ratio** | $\frac{\text{Exact Matches}}{\text{Total Shadow Comparisons}}$ | $> 85.0\%$ | Primary and Phase 6D extract identical interests, locations, and budgets. |
| **Partial Match Ratio**| $\frac{\text{Partial Matches}}{\text{Total Shadow Comparisons}}$ | $< 12.0\%$ | Semantic overlap (e.g., "pottery" vs "artisanal pottery") with no contradictions. |
| **Hard Disagreement** | $\frac{\text{Disagreements}}{\text{Total Shadow Comparisons}}$ | $< 3.0\%$ | Contradictory extractions (e.g., different budget numbers or contradictory locations). **Alert if $> 5\%$.** |
| **Placeholder Emergence**| $\frac{\text{Hallucinated Placeholders}}{\text{Total Shadow Comparisons}}$ | **$0.0\%$ (Strict Zero)** | Emergence of `city_a`, `city_b`, `N/A`, or unstated budgets. **Critical Alert if $> 0\%$.** |

---

## 7. Production Safety Gate Preservation & Verification

### 7.1 Multi-Layer Architecture
The production safety gate is embedded in two independent layers:
1. **Layer 1: Configuration Gate ([`src/config/environment.ts`](file:///Users/shivamsingh/Desktop/foryou/src/config/environment.ts#L95-L104)):**
   ```typescript
   if (aiEnv === 'production') {
     if (process.env.STAGING_PHASE6D === 'true' || process.env.USE_PHASE6D_ADAPTER === 'true' || process.env.AI_PROVIDER === 'phase6d') {
       throw new Error(
         '[CRITICAL SAFETY GATE] Phase 6D local adapter cannot be activated in AI_ENV=production. ' +
         'Production strictly requires validated cloud providers (Gemini / OpenAI).'
       );
     }
   }
   ```
2. **Layer 2: Provider Resolution Gate ([`src/ai/provider/llmProvider.ts`](file:///Users/shivamsingh/Desktop/foryou/src/ai/provider/llmProvider.ts#L630-L644)):**
   - In `getActiveLlmClient()`, if `config.aiEnv === 'production'`, the function unconditionally ignores staging flags and exclusively initializes `ResilientLlmOrchestrator(primary, secondary)` (`gemini-3.5-flash-lite` with fallback to `gpt-4o-mini`).

### 7.2 Verification Evidence
The gate was tested live by executing `scratch/verify_gate.py`:
- Injected condition: `AI_ENV=production`, `STAGING_PHASE6D=true`.
- Result: **Execution terminated with Return Code 0 (Passed Gate).**
- Observed error output:
  ```text
  [CRITICAL SAFETY GATE] Phase 6D local adapter cannot be activated in AI_ENV=production.
  Production strictly requires validated cloud providers (Gemini / OpenAI).
  ```

---

## 8. Phased Implementation Roadmap & Pre-Canary Checklist

```
+---------------------------------------------------------------------------------------+
| PHASE 1: LOCAL STAGING LOAD TESTING                                                   |
| • Execute k6/Locust load tests on candidate containerized serving architecture.        |
| • Verify queue backpressure (503s), timeouts (504s), and zero process crashes.        |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| PHASE 2: CLOUD SERVING INFRASTRUCTURE DEPLOYMENT                                      |
| • Deploy vLLM (Linux GPU) or llama-server (Linux CPU) on isolated internal VPC.       |
| • Expose health checks (/health/ready) and metrics (/metrics) to Prometheus.          |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| PHASE 3: PRODUCTION SHADOW-TRAFFIC EVALUATION                                         |
| • Enable ShadowMemoryExtractor in production (SHADOW_PHASE6D_ENABLED=true).           |
| • Run 100% read-only shadow comparison against live Gemini extractions for 7 days.     |
| • Target: >85% exact match, <3% disagreement, exactly 0% placeholder emergence.       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| PHASE 4: LIMITED PRODUCTION CANARY (CONDITIONAL)                                      |
| • Route 1% of live memory_extraction operations to Phase 6D with automated failover.  |
| • Zero live routing for conversational onboarding or activity planning.              |
+---------------------------------------------------------------------------------------+
```

### Pre-Canary Go / No-Go Checklist
- [ ] **Infrastructure:** Serving container running vLLM or llama-server with continuous batching deployed on private cloud VPC.
- [ ] **Load Test:** 15-minute 50-user load test completed with $p_{95} < 5,000\text{ ms}$ and error rate $< 0.5\%$.
- [ ] **Shadow Evaluation:** Minimum 1,000 live production shadow comparisons recorded.
- [ ] **Disagreement Rate:** Hard disagreement rate $< 3.0\%$ with zero synthetic placeholders.
- [ ] **Safety Isolation:** Confirmed zero calls to `MemoryManager.reconcileAndSave()` from shadow pipeline.
- [ ] **Fail-Fast Verification:** Re-verify that disabling shadow mode returns the system to 100% cloud execution with zero lingering dependencies.
