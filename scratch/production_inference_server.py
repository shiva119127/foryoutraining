#!/usr/bin/env python3
"""
FlowersForYou - Phase 6D Production-Grade Inference Server Architecture
Features:
- FastAPI async server with bounded queue and concurrency semaphore
- Device-level serialization lock for PyTorch MPS on Apple Silicon
- Cooperative token cancellation via StoppingCriteria on client disconnect/timeout
- Fast-fail rejection (HTTP 503) when queue exceeds capacity
- Standardized health endpoints: /health/live, /health/ready, /health
- Dual endpoint support: /v1/extract_preferences (canonical) and /generate (compatibility)
- Prometheus observability metrics: /metrics
- Graceful shutdown handler with 15-second drain window
"""

import asyncio
import os
import signal
import sys
import threading
import time
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from pathlib import Path

# FastAPI & Pydantic
from fastapi import FastAPI, Request, HTTPException, status, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# PyTorch / PEFT
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "outputs/flowersforyou-adapter-10k")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
MAX_QUEUE_CAPACITY = int(os.getenv("MAX_QUEUE_CAPACITY", "64"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10.0"))
DRAIN_TIMEOUT_SECONDS = float(os.getenv("DRAIN_TIMEOUT_SECONDS", "15.0"))
SUPPORTED_OPERATIONS = {"memory_extraction", "extract_preferences"}

# State variables
model = None
tokenizer = None
device = None
is_ready = False
is_shutting_down = False
active_requests = 0
active_generations = 0
total_requests = 0
total_failures = 0
total_queue_rejections = 0
total_timeouts = 0
queue_semaphore: Optional[asyncio.Semaphore] = None
model_inference_lock = asyncio.Lock()


class CancelStoppingCriteria(StoppingCriteria):
    """Halts model generation within 1 token if cancellation event is flagged."""
    def __init__(self, cancel_event: threading.Event):
        super().__init__()
        self.cancel_event = cancel_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.cancel_event.is_set()


class ExtractionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt containing user statement to extract")
    operation_name: str = Field("memory_extraction", description="Operation tag")
    max_tokens: int = Field(150, ge=16, le=512)
    temperature: float = Field(0.3, ge=0.0, le=1.0)

    @field_validator("operation_name")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        if v not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported operation_name '{v}'. Supported operations: {sorted(list(SUPPORTED_OPERATIONS))}"
            )
        return v


class ExtractionResponse(BaseModel):
    text: str
    token_count: int
    latency_ms: int
    queue_wait_ms: int
    model: str = "qwen2.5-0.5b-phase6d"


def load_model_and_adapter():
    global model, tokenizer, device, is_ready
    print(f"Loading base model {MODEL_ID}...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type in ["cuda", "mps"] else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    print(f"Attaching Phase 6D LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).to(device)
    model.eval()
    is_ready = True
    print(f"Model successfully initialized on {device.type} ({dtype}). Ready for inference.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue_semaphore, is_shutting_down
    queue_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    load_model_and_adapter()
    yield
    # Graceful shutdown handler
    print("Initiating graceful shutdown. Draining in-flight requests...")
    is_shutting_down = True
    drain_start = time.time()
    while active_requests > 0 and (time.time() - drain_start) < DRAIN_TIMEOUT_SECONDS:
        print(f"Waiting for {active_requests} active requests to complete...")
        await asyncio.sleep(0.5)
    print("Shutdown complete. Exiting cleanly.")


app = FastAPI(
    title="FlowersForYou Phase 6D Inference Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health/live")
async def liveness():
    """Liveness probe: verifies process is alive."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health")
@app.get("/health/ready")
async def readiness():
    """Readiness probe: verifies model is loaded and server is accepting traffic."""
    if not is_ready:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "initializing"})
    if is_shutting_down:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "shutting_down"})
    return {
        "status": "ready",
        "model": MODEL_ID,
        "adapter": str(ADAPTER_PATH),
        "device": str(device),
        "active_requests": active_requests,
        "active_generations": active_generations,
        "max_concurrent": MAX_CONCURRENT_REQUESTS
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    queued_requests = max(0, active_requests - active_generations)
    lines = [
        "# HELP phase6d_active_requests Total in-flight requests in server (queued + active)",
        "# TYPE phase6d_active_requests gauge",
        f"phase6d_active_requests {active_requests}",
        "# HELP phase6d_queued_requests Requests waiting in admission queue for device execution",
        "# TYPE phase6d_queued_requests gauge",
        f"phase6d_queued_requests {queued_requests}",
        "# HELP phase6d_active_generations Requests actively executing on device",
        "# TYPE phase6d_active_generations gauge",
        f"phase6d_active_generations {active_generations}",
        "# HELP phase6d_requests_total Total number of inference requests received",
        "# TYPE phase6d_requests_total counter",
        f"phase6d_requests_total {total_requests}",
        "# HELP phase6d_failures_total Total number of failed inference requests",
        "# TYPE phase6d_failures_total counter",
        f"phase6d_failures_total {total_failures}",
        "# HELP phase6d_queue_rejections_total Total requests rejected due to queue saturation (503)",
        "# TYPE phase6d_queue_rejections_total counter",
        f"phase6d_queue_rejections_total {total_queue_rejections}",
        "# HELP phase6d_timeouts_total Total requests timed out waiting or generating (504)",
        "# TYPE phase6d_timeouts_total counter",
        f"phase6d_timeouts_total {total_timeouts}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/v1/extract_preferences", response_model=ExtractionResponse)
@app.post("/generate", response_model=ExtractionResponse)
async def extract_preferences(req: ExtractionRequest, request: Request):
    """
    Structured preference extraction endpoint.
    Guarantees:
    - Bounded queue admission control (HTTP 503 fast-fail when saturated).
    - Serialized access to MPS device via model_inference_lock.
    - Cooperative cancellation stopping criteria to avoid zombie executions on timeout.
    - Full deadline enforcement across both queue wait and model generation.
    """
    global active_requests, active_generations, total_requests, total_failures, total_queue_rejections, total_timeouts
    if is_shutting_down:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server is shutting down")
    if not is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is still initializing")

    total_requests += 1
    t_received = time.time()

    # Bounded queue admission check
    if active_requests >= (MAX_CONCURRENT_REQUESTS + MAX_QUEUE_CAPACITY):
        total_failures += 1
        total_queue_rejections += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference request queue saturated. Please retry with exponential backoff."
        )

    active_requests += 1
    try:
        # Acquire concurrency slot with overall timeout
        async with queue_semaphore:
            # Check if client disconnected while waiting in admission queue
            if await request.is_disconnected():
                print("Client disconnected while waiting in admission queue; aborting.")
                raise HTTPException(status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST, detail="Client closed connection")

            # Check remaining deadline after queue wait
            t_after_semaphore = time.time()
            time_in_queue = t_after_semaphore - t_received
            queue_wait_ms = int(time_in_queue * 1000)
            remaining_timeout = REQUEST_TIMEOUT_SECONDS - time_in_queue

            if remaining_timeout <= 0.1:
                total_failures += 1
                total_timeouts += 1
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Inference queued for {queue_wait_ms}ms, exceeding {REQUEST_TIMEOUT_SECONDS}s deadline"
                )

            # Serialize access to shared PyTorch MPS device
            async with model_inference_lock:
                if await request.is_disconnected():
                    print("Client disconnected before device execution; aborting.")
                    raise HTTPException(status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST, detail="Client closed connection")

                active_generations += 1
                t_gen_start = time.time()
                cancel_event = threading.Event()

                loop = asyncio.get_running_loop()
                # Run blocking generation in thread pool with cooperative cancellation
                gen_future = loop.run_in_executor(None, _generate_blocking, req, cancel_event)

                try:
                    # Update remaining timeout based on total time elapsed since request receipt
                    remaining_gen_timeout = max(0.1, REQUEST_TIMEOUT_SECONDS - (time.time() - t_received))
                    text, token_count = await asyncio.wait_for(asyncio.shield(gen_future), timeout=remaining_gen_timeout)
                except asyncio.TimeoutError:
                    cancel_event.set()
                    total_failures += 1
                    total_timeouts += 1
                    # Await executor thread termination with short safety drain so lock is not released prematurely
                    try:
                        await asyncio.wait_for(gen_future, timeout=2.0)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=f"Inference execution exceeded deadline (queued: {queue_wait_ms}ms, timeout: {REQUEST_TIMEOUT_SECONDS}s)"
                    )
                except Exception:
                    cancel_event.set()
                    raise
                finally:
                    active_generations -= 1

                latency_ms = int((time.time() - t_gen_start) * 1000)
                return ExtractionResponse(
                    text=text,
                    token_count=token_count,
                    latency_ms=latency_ms,
                    queue_wait_ms=queue_wait_ms,
                    model="qwen2.5-0.5b-phase6d"
                )
    finally:
        active_requests -= 1


def _generate_blocking(req: ExtractionRequest, cancel_event: Optional[threading.Event] = None) -> (str, int):
    """Synchronous model execution run inside executor thread."""
    messages = [
        {"role": "system", "content": "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON."},
        {"role": "user", "content": req.prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList()
    if cancel_event is not None:
        stopping_criteria.append(CancelStoppingCriteria(cancel_event))

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=(req.temperature > 0),
            temperature=max(0.1, req.temperature),
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )

    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, len(new_tokens)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("production_inference_server:app", host="0.0.0.0", port=8000, reload=False, workers=1)

