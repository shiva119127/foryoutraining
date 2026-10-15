#!/usr/bin/env python3
"""
FlowersForYou - Phase 6D Staging Inference Server
A multithreaded HTTP server that serves the Phase 6D fine-tuned adapter
for staging validation and automated performance profiling.
"""

import json
import os
import re
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs/flowersforyou-adapter-10k"
PORT = 8765

model = None
tokenizer = None
device = None
startup_latency_ms = 0
gpu_lock = threading.Lock()

SYSTEM_PERSONAS = {
    "memory_extraction": (
        "You are the FlowersForYou preference extractor. Extract stated preferences as valid JSON and do not invent details."
    ),
    "onboarding_reasoning": (
        "You are the FlowersForYou onboarding assistant. Learn preferences naturally. "
        "Ask one relevant question at a time. Do not turn onboarding into a rigid survey."
    ),
    "plan_generation": (
        "You are the FlowersForYou planner. Onboarding is complete. "
        "Use the available profile and ask only essential clarifying questions."
    ),
    "returning_greeting": (
        "You are the FlowersForYou assistant. If a returning user has completed onboarding, "
        "do not repeat onboarding. Greet them warmly and invite a new plan."
    ),
}


def load_staging_model():
    global model, tokenizer, device, startup_latency_ms
    t0 = time.time()
    print(f"Loading Tokenizer & Model ({MODEL_ID}) for Staging...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    print(f"Attaching Staging Adapter: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH)).to(device)
    model.eval()

    startup_latency_ms = int((time.time() - t0) * 1000)
    print(f"Phase 6D Staging Model loaded successfully in {startup_latency_ms}ms on {device.type}.")


class StagingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "ready",
                "environment": "staging",
                "model_id": MODEL_ID,
                "adapter": str(ADAPTER_PATH.name),
                "device": str(device),
                "startup_latency_ms": startup_latency_ms,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/generate":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            # Reliability simulation hooks
            sim = data.get("simulate_error")
            if sim == "timeout":
                time.sleep(20)  # Exceeds client timeout
                self.send_response(504)
                self.end_headers()
                self.wfile.write(b'{"error": "Gateway Timeout"}')
                return
            elif sim == "503":
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error": "Service Unavailable"}')
                return
            elif sim == "malformed_json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"text": "```json\n{ action: invalid, budget: [bad] \n```", "latency_ms": 10}).encode("utf-8"))
                return
            elif sim == "non_json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"text": "I am unable to output JSON right now.", "latency_ms": 10}).encode("utf-8"))
                return

            prompt_text = data.get("prompt", "")
            messages = data.get("messages")
            op_name = data.get("operation_name", "memory_extraction")
            max_tokens = int(data.get("max_tokens", 150))
            temperature = float(data.get("temperature", 0.3))

            t_start = time.time()
            if not messages:
                sys_prompt = SYSTEM_PERSONAS.get(op_name, SYSTEM_PERSONAS["memory_extraction"])
                # Extract clean user utterance if prompt wrapped with guidelines
                user_msg = prompt_text
                if 'Current User Message to Extract:' in prompt_text:
                    m = re.search(r'Current User Message to Extract:\s*"([^"]+)"', prompt_text)
                    if m:
                        user_msg = m.group(1)
                elif 'Current User Input:' in prompt_text:
                    m = re.search(r'Current User Input:\s*"([^"]+)"', prompt_text)
                    if m:
                        user_msg = m.group(1)

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ]

            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with gpu_lock:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=(temperature > 0),
                        temperature=max(0.1, temperature),
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            new_tokens = out[0][input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            inference_ms = int((time.time() - t_start) * 1000)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "text": text,
                "token_count": len(new_tokens),
                "latency_ms": inference_ms,
                "model": "qwen2.5-0.5b-phase6d"
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    load_staging_model()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), StagingHandler)
    print(f"FlowersForYou Staging Inference Server listening on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    run_server()
