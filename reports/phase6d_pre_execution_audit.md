# FlowersForYou Phase 6D — Pre-Execution Infrastructure Audit Report

**Audit Type:** Pre-Execution Infrastructure, Concurrency, and Shadow-Testing Audit  
**Date:** October 16, 2026  
**Audited Artifacts:**
- Plan: `reports/phase6d_inference_infrastructure_plan.md`
- Server Prototype: `scratch/production_inference_server.py`
- Shadow Module: `scratch/shadow_memory_extractor.ts`
- Load Test Scripts: `scratch/load_test_phase6d.js` & `scratch/locustfile.py`
- Safety Gate Test: `scratch/verify_gate.py`  
**Host Application:** `~/Desktop/foryou`  
**Fine-Tuning Repository:** `~/Downloads/flowersforyou-finetuning`

---

## Executive Summary & Decision

### **EXECUTIVE DECISION: NEEDS FIXES**
**Current Status: BLOCKED for Full 15-Minute Load Test; CONDITIONAL PASS for Small 5-User Smoke Test (Pending Two Route & Concurrency Fixes).**

This audit performed a line-by-line inspection of all serving infrastructure, load-testing scripts, shadow extraction modules, and safety gates created in preparation for Phase 6D inference.

### Key Audit Findings
1. **Critical Path Mismatch ([CRITICAL-01]):** `production_inference_server.py` defines route `@app.post("/v1/extract_preferences")`, while all clients (`load_test_phase6d.js`, `locustfile.py`, `shadow_memory_extractor.ts`, and `staging_inference_server.py`) send requests to `POST /generate`. Running the load test or shadow module against the new server prototype produces **HTTP 404 on 100% of requests**.
2. **Unsafe MPS Concurrency ([CRITICAL-02]):** `production_inference_server.py` declares `model_inference_lock = asyncio.Lock()` at line 50, but **never acquires or references it**. Instead, it executes `model.generate()` concurrently in Python's `ThreadPoolExecutor` across up to 8 threads on a shared PyTorch model. PyTorch Metal Performance Shaders (MPS) on Apple Silicon is not re-entrant or thread-safe for concurrent forward passes on the same module, risking memory corruption or segfaults under concurrent load.
3. **Zombie Execution on Client Timeout ([CRITICAL-03]):** `asyncio.wait_for(_run_generation(), timeout=10.0)` cancels the asyncio task when timed out, but **cannot cancel the underlying OS thread** running `_generate_blocking` in `ThreadPoolExecutor`. The semaphore slot is released while the zombie thread continues consuming GPU/CPU compute, exacerbating overload.
4. **Synchronous File I/O in Shadow Mode ([HIGH-01]):** `shadow_memory_extractor.ts` uses `fs.appendFileSync()` on line 213. Even though deferred via `setImmediate()`, this synchronous file write runs on the main Node.js event loop, introducing event-loop blocking into live production customer traffic.
5. **Unrealistic k6 Thresholds ([HIGH-02]):** k6 enforces $p_{50} < 3,000\text{ ms}$ under 50 concurrent virtual users. On a single local worker with zero load, warm median generation latency is already 2,990 ms. Under concurrency, queuing delay guarantees this threshold will fail, producing false test failures.

---

## 1. Audit Findings Ranked by Severity

### Critical Findings

#### [CRITICAL-01] Endpoint Path Mismatch Between Server Prototype and Clients
- **Files Involved:**
  - `scratch/production_inference_server.py:L149` (`@app.post("/v1/extract_preferences")`)
  - `scratch/load_test_phase6d.js:L45` (`TARGET_ENDPOINT = ... || 'http://127.0.0.1:8765/generate'`)
  - `scratch/locustfile.py:L62` (`self.client.post("/generate", ...)`)
  - `scratch/shadow_memory_extractor.ts:L82` (`fetch(`${this.endpoint}/generate`, ...)`)
- **Analysis:** The new FastAPI server implementation exposes `/v1/extract_preferences`. Both load test scripts and the shadow module are hardcoded to `/generate`. Furthermore, k6 defaults to port `8765` (the old staging server port) rather than `8000` (the FastAPI port).
- **Impact:** Running k6 or Locust against `http://127.0.0.1:8000` returns HTTP 404 on every request.
- **Proposed Minimal Fix:** In `production_inference_server.py`, add route aliases `@app.post("/generate")` and `@app.get("/health")` to maintain 100% backward compatibility with all test scripts and shadow clients.

#### [CRITICAL-02] Unsafe Multi-Threaded Model Execution on Shared PyTorch MPS Instance
- **Files Involved:**
  - `scratch/production_inference_server.py:L50, L171, L182-L184, L217-L225`
- **Analysis:** Line 50 declares `model_inference_lock = asyncio.Lock()`, but the variable is never used in the request handler. Line 171 acquires `async with queue_semaphore:`, allowing up to 8 concurrent requests into `loop.run_in_executor(None, _generate_blocking, req)`. In Python, this runs `model.generate()` concurrently across multiple OS worker threads on the exact same `nn.Module` and device. PyTorch's MPS (Metal) backend on macOS is not thread-safe for concurrent generation forward passes.
- **Impact:** Concurrency $> 1$ on local Apple Silicon staging risks race conditions in Metal command buffers, corrupted KV-cache allocations, or process crashes.
- **Proposed Minimal Fix:** On local Apple Silicon staging, wrap generation with `async with model_inference_lock:` so that admissions can queue safely while model execution remains serialized per device. For Linux GPU production, use an engine built for concurrent batching (vLLM).

#### [CRITICAL-03] Zombie Thread Execution on Timeout and Semaphore Leakage
- **Files Involved:**
  - `scratch/production_inference_server.py:L171-L204`
- **Analysis:** `asyncio.wait_for(_run_generation(), timeout=REQUEST_TIMEOUT_SECONDS)` cancels the asyncio task when the 10-second deadline expires. However, Python's `ThreadPoolExecutor` cannot interrupt a running thread. The thread running `_generate_blocking` continues computing tokens in the background until completion. Furthermore, the `async with queue_semaphore:` context exits upon `TimeoutError`, releasing the concurrency slot immediately.
- **Impact:** A timed-out request continues burning GPU/CPU cycles in a zombie thread while the server admits a new request into the concurrency slot, compounding server saturation.
- **Proposed Minimal Fix:** Implement a cooperative cancellation check or stopping criteria in `model.generate()`, and do not release the underlying device semaphore until the executor thread has actually exited.

---

### High Findings

#### [HIGH-01] Synchronous Disk I/O (`fs.appendFileSync`) in Node.js Shadow Pipeline
- **Files Involved:**
  - `scratch/shadow_memory_extractor.ts:L213`
- **Analysis:** Line 213 uses `fs.appendFileSync(logFile, JSON.stringify(log) + '\n', 'utf-8')`. While `triggerShadowEvaluation` uses `setImmediate()` to defer invocation, the callback executes on the main Node.js event loop thread. `appendFileSync` performs blocking synchronous filesystem I/O.
- **Impact:** Under concurrent production traffic (e.g., 30–50 requests/sec), synchronous disk operations block the Node.js event loop, adding unpredictable jitter and latency to live user requests.
- **Proposed Minimal Fix:** Replace `fs.appendFileSync` with asynchronous `await fs.promises.appendFile(logFile, ...)`.

#### [HIGH-02] Unsupported Latency Thresholds in k6 Load Test Script
- **Files Involved:**
  - `scratch/load_test_phase6d.js:L22-L23`
- **Analysis:** The script enforces `'http_req_duration': ['p(50)<3000', 'p(95)<5000', 'p(99)<7500']`. Our empirical baseline audit established that a single request with zero load takes **2,990 ms** on Apple Silicon MPS. Under 50 concurrent virtual users, queuing delay alone will push total response times past 3,000 ms.
- **Impact:** The k6 script will report failure on thresholds even when the inference server performs correctly within hardware constraints.
- **Proposed Minimal Fix:** Re-label latency thresholds as experimental targets rather than hard SLAs. For local staging, configure realistic thresholds ($p_{50} < 8,000\text{ ms}$, $p_{95} < 14,000\text{ ms}$).

#### [HIGH-03] Conflation of Virtual Users (VUs) and In-Flight Request Concurrency
- **Files Involved:**
  - `reports/phase6d_inference_infrastructure_plan.md:Section 5`
  - `scratch/load_test_phase6d.js:L95` (`sleep(Math.random() * 2 + 1)`)
  - `scratch/locustfile.py:L28` (`wait_time = between(1.0, 3.0)`)
- **Analysis:** The plan claims the test generates "50 concurrent in-flight requests." However, both scripts incorporate realistic user think time (1 to 3 seconds). With an average think time of 2s and an inference duration of ~3s, each cycle takes 5s. Thus, 50 VUs generate an average of $\frac{3}{5} \times 50 \approx 30$ simultaneous in-flight requests.
- **Impact:** Mismatch between expected queue depth and observed load.
- **Proposed Minimal Fix:** Document that 50 VUs with think time models an arrival rate of ~10 req/s and an average concurrency of ~30 in-flight requests.

---

### Medium Findings

#### [MEDIUM-01] Queue Waiting Time Is Excluded from Server Execution Timeout
- **Files Involved:**
  - `scratch/production_inference_server.py:L171-L188`
- **Analysis:** `REQUEST_TIMEOUT_SECONDS = 10.0` is applied only inside `asyncio.wait_for(_run_generation(), timeout=10.0)`. Time spent waiting in the semaphore queue (`queue_wait_ms`) is not bounded by this timeout. A request can wait 15 seconds in queue, then run 9 seconds in generation, taking 24 seconds total.
- **Impact:** Clients with an 18s timeout (k6/Locust) or an 8s timeout (shadow module) will disconnect before the server finishes, wasting server resources.
- **Proposed Minimal Fix:** Enforce an overall deadline from time of admission: `remaining = max(0.1, REQUEST_TIMEOUT_SECONDS - queue_wait)`. If `queue_wait >= REQUEST_TIMEOUT_SECONDS`, reject immediately without executing.

#### [MEDIUM-02] Privacy & Pseudonymization Deficiencies in Shadow Logs
- **Files Involved:**
  - `scratch/shadow_memory_extractor.ts:L126, L182, L206-L214`
- **Analysis:**
  a) User IDs are hashed using an unsalted SHA-256 (`crypto.createHash('sha256').update(userId).digest('hex').slice(0, 16)`). Unsalted hashes are vulnerable to dictionary/rainbow-table re-identification.
  b) Plaintext preference values (interests, locations, budgets) and raw model text are logged to disk.
  c) No file retention, rotation, or restricted permissions (`0600`) are enforced.
- **Impact:** Violates data minimization principles for live shadow evaluation.
- **Proposed Minimal Fix:** Use HMAC-SHA256 with an environment salt (`SHADOW_LOG_SALT`), set file permissions to `0600`, and establish a 14-day log retention script.

#### [MEDIUM-03] Metric Conflation in Prometheus Endpoint
- **Files Involved:**
  - `scratch/production_inference_server.py:L138`
- **Analysis:** `phase6d_active_requests` is incremented on line 168 (before acquiring the semaphore) and decremented in `finally`. It therefore represents the sum of **queued requests + actively generating requests**.
- **Impact:** Prometheus dashboards cannot distinguish between queue saturation and active model utilization.
- **Proposed Minimal Fix:** Expose two distinct gauges: `phase6d_active_generations` (inside semaphore) and `phase6d_queued_requests` (waiting for semaphore).

#### [MEDIUM-04] Safety Gate Test Exercises Configuration Only, Not Full Provider Dispatch
- **Files Involved:**
  - `scratch/verify_gate.py:L10-L15`
- **Analysis:** `verify_gate.py` imports `src/config/environment.ts` and calls `loadAndValidateConfig()`. It proves that `environment.ts` throws when `AI_ENV=production` and `STAGING_PHASE6D=true`. However, it does not import `src/ai/provider/llmProvider.ts` or call `getActiveLlmClient()` to verify runtime provider dispatch fallback.
- **Impact:** While the configuration gate is confirmed, provider resolution in `llmProvider.ts` is not covered by this script.
- **Proposed Minimal Fix:** Update `verify_gate.py` to also invoke `getActiveLlmClient()` and assert that it throws or returns `ResilientLlmOrchestrator`.

---

### Low Findings

#### [LOW-01] Untested Assumption Regarding llama.cpp GGUF Conversion
- **Files Involved:**
  - `reports/phase6d_inference_infrastructure_plan.md:Section 2.2`
- **Analysis:** The plan recommends `llama.cpp` for CPU serving, but no conversion script (`merge_and_unload()` + `convert_hf_to_gguf.py`) has been executed or verified.
- **Proposed Minimal Fix:** Provide a verified one-click conversion script (`scratch/convert_phase6d_to_gguf.py`) before attempting `llama-server` deployment.

#### [LOW-02] k6 Error Rate Metric Does Not Distinguish Expected 503 Rejections from 500 Crashes
- **Files Involved:**
  - `scratch/load_test_phase6d.js:L73`
- **Analysis:** `errorRate.add(!isSuccess)` increments for any non-200 response, penalizing expected 503 backpressure rejections identically to internal server errors (500).
- **Proposed Minimal Fix:** Track `admission_rejection_rate` (503/429) separately from `server_error_rate` (500/502/504).

---

## 2. Load-Test Endpoint & Payload Mapping Table

The following table maps every proposed load-test endpoint and payload against the actual server implementation:

| Client / Test Script | Target URL / Host | HTTP Method | Request Payload Format | Response Payload Format | Actual Server Endpoint Match? | Status / Finding |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **`scratch/load_test_phase6d.js` (k6)** | Default: `http://127.0.0.1:8765/generate`<br>Option: `http://127.0.0.1:8000/generate` | `POST` | `{"prompt": string, "operation_name": string, "temperature": float, "max_tokens": int}` | `{"text": string, "token_count": int, "latency_ms": int, ...}` | Matches `staging_inference_server.py:8765`.<br>**MISMATCHES `production_inference_server.py:8000`** | **CRITICAL MISMATCH:** If run against `:8000`, returns HTTP 404. |
| **`scratch/locustfile.py` (Locust)** | `{host}/generate` (CLI defaults to `:8765`) | `POST` | `{"prompt": string, "operation_name": string, "temperature": float, "max_tokens": int}` | `{"text": string, "token_count": int, "latency_ms": int, ...}` | Matches `staging_inference_server.py:8765`.<br>**MISMATCHES `production_inference_server.py:8000`** | **CRITICAL MISMATCH:** Hardcoded `/generate` path fails with HTTP 404 on port 8000. |
| **`scratch/shadow_memory_extractor.ts`** | Default: `http://127.0.0.1:8765/generate` | `POST` | `{"prompt": string, "operation_name": string, "temperature": float, "max_tokens": int}` | `{"text": string, ...}` | Matches `staging_inference_server.py:8765`.<br>**MISMATCHES `production_inference_server.py:8000`** | **MISMATCH:** Shadow client cannot communicate with production prototype on port 8000. |
| **`scratch/staging_inference_server.py`** | `http://127.0.0.1:8765/generate` | `POST` | Untyped JSON (`prompt`, `operation_name`, `max_tokens`, `temperature`) | `{"text": string, "token_count": int, "latency_ms": int, "model": string}` | Implements `POST /generate` and `GET /health` | Working in staging, but serializes under `gpu_lock`. |
| **`scratch/production_inference_server.py`** | `http://0.0.0.0:8000/v1/extract_preferences` | `POST` | Pydantic `ExtractionRequest` (`prompt`, `operation_name`, `max_tokens`, `temperature`) | Pydantic `ExtractionResponse` (`text`, `token_count`, `latency_ms`, `queue_wait_ms`, `model`) | Implements `POST /v1/extract_preferences`, `GET /health/live`, `GET /health/ready`, `GET /metrics` | **ISOLATED:** Missing `/generate` alias. |

---

## 3. Verified Facts vs. Assumptions vs. Untested Behavior

### 3.1 Verified Facts (Empirically Confirmed)
- Checkpoint 625 matches `outputs/flowersforyou-adapter-10k` with 100% SHA-256 identity (`216e076d6cb3...`).
- `AI_ENV=production STAGING_PHASE6D=true` trips the fail-fast safety gate in `src/config/environment.ts` with Return Code 0.
- Single-request warm generation latency on Apple Silicon MPS is **2,990 ms** with a cold start of **5,612 ms**.
- `staging_inference_server.py` uses `gpu_lock = threading.Lock()` around `model.generate()`.
- `shadow_memory_extractor.ts` does not call `MemoryManager.reconcileAndSave()`.

### 3.2 Assumptions (Unvalidated Hypotheses)
- That vLLM can be deployed on target cloud instances without testing container build, PagedAttention kernels, and LoRA loading.
- That `llama-server` can load Phase 6D without testing GGUF weight conversion.
- That 50 concurrent virtual users can achieve $p_{50} < 3,000\text{ ms}$ on local staging hardware.
- That unsalted SHA-256 slicing provides GDPR/SOC-2 compliant anonymization.

### 3.3 Untested Behavior (Known Gaps)
- `production_inference_server.py` has never been executed or received a live HTTP request.
- The queue admission control (rejecting at $>72$ in-flight requests) has never been tested under load.
- Client disconnect cancellation (`await request.is_disconnected()`) has not been tested against dropped TCP connections.
- Shadow mode has never been wired into the live Next.js request pipeline.

---

## 4. Prioritized Remediation List (Minimal Proposed Fixes)

Do not apply these fixes automatically. These are proposed for review:

1. **Fix Critical-01 (Route Aliasing in `production_inference_server.py`):**
   Add route decorators so that `production_inference_server.py` answers on both `/generate` and `/v1/extract_preferences`, and `/health` as well as `/health/ready`:
   ```python
   @app.post("/generate", response_model=ExtractionResponse)
   @app.post("/v1/extract_preferences", response_model=ExtractionResponse)
   async def extract_preferences(req: ExtractionRequest, request: Request):
   ```
2. **Fix Critical-02 (Device Thread-Safety on MPS in `production_inference_server.py`):**
   Wrap `_run_generation()` with `async with model_inference_lock:` so that admissions can queue asynchronously while MPS forward passes remain serialized per device:
   ```python
   async with queue_semaphore:
       async with model_inference_lock:
           text, token_count = await _run_generation()
   ```
3. **Fix High-01 (Async File I/O in `shadow_memory_extractor.ts`):**
   Replace synchronous file write with asynchronous non-blocking I/O:
   ```typescript
   await fs.promises.appendFile(logFile, JSON.stringify(log) + '\n', 'utf-8');
   ```
4. **Fix Medium-01 (Overall Request Deadline in `production_inference_server.py`):**
   Enforce total deadline from admission time:
   ```python
   time_in_queue = time.time() - t_received
   remaining_timeout = max(0.1, REQUEST_TIMEOUT_SECONDS - time_in_queue)
   if remaining_timeout <= 0.1:
       raise HTTPException(status_code=504, detail="Request timed out waiting in admission queue")
   ```
5. **Fix High-02 & Low-02 (Update k6 Script Thresholds):**
   Adjust k6 thresholds for local staging ($p_{50} < 8,000\text{ ms}$, $p_{95} < 14,000\text{ ms}$) and separate 503 rejections from 500 server crashes.

---

## 5. Safe, Exact Sequence for Local Smoke Test and Full Load Test

Follow this exact sequence. **Do not run the full 15-minute load test until Steps 1–3 pass completely.**

### Step 1: Apply Minimal Code Fixes (Pre-requisite, upon approval)
- Add `/generate` route alias to `scratch/production_inference_server.py`.
- Wrap generation in `model_inference_lock`.

### Step 2: Start Server Prototype Locally
```bash
./.venv/bin/python scratch/production_inference_server.py
```
*Verify output: `Model successfully initialized on mps. Ready for inference.` Listening on `http://0.0.0.0:8000`.*

### Step 3: Run Single-Request Smoke Test (Manual Verification)
```bash
# 1. Verify health endpoints
curl -s http://127.0.0.1:8000/health/live
curl -s http://127.0.0.1:8000/health/ready

# 2. Verify extraction via /generate
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "User message: I like pottery in Indiranagar under 500."}'

# 3. Verify metrics
curl -s http://127.0.0.1:8000/metrics
```
*Pass Criteria: HTTP 200 returned, valid JSON extracted, metrics incremented.*

### Step 4: Run Micro Concurrency Smoke Test (5 VUs for 30 Seconds)
```bash
k6 run --vus 5 --duration 30s scratch/load_test_phase6d.js --env TARGET_URL=http://127.0.0.1:8000/generate
```
*Pass Criteria: Zero 404s, zero 500 crashes, valid JSON rate $> 98\%$.*

### Step 5: Execute Full Staging Load Test (50 VUs for 15 Minutes)
*(Execute ONLY after Step 4 passes cleanly).*
```bash
k6 run scratch/load_test_phase6d.js --env TARGET_URL=http://127.0.0.1:8000/generate
```

---

## 6. Audit Confirmation & Non-Negotiable Compliance

I confirm explicitly:
- **Zero production deployments** or configuration mutations were performed.
- **Zero live customer traffic** was routed to the local daemon.
- **The cloud provider remains the active production default.**
- **No model weights, adapters, training data, or evaluation datasets** were modified.
- **Zero shadow outputs** were written to user memory or production databases.
- **The full 15-minute load test was NOT executed.**
- All findings are backed by direct inspection of source code and configuration files.
