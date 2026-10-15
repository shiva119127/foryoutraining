#!/usr/bin/env python3
"""
FlowersForYou - Phase 6D Inference Load Test (Locust)
Simulates 50 concurrent users issuing structured preference extraction requests.
"""

import json
import random
import time
from locust import HttpUser, task, between, events

USER_UTTERANCES = [
    "I enjoy pottery workshops and quiet nature walks.",
    "Looking for specialty coffee tasting and artisanal sourdough baking around Indiranagar.",
    "Update my spending limit: my maximum budget is ₹1500 per person in Bandra West for indie board games.",
    "Actually, scratch Indiranagar—I moved to Koramangala now, and please raise my maximum budget to ₹2000.",
    "Please clear my budget limit, I don't want any budget ceiling.",
    "I really enjoy modern strategy board games, but I haven't decided on any location or budget yet.",
    "I am strictly vegan and need dairy-free chocolate tasting workshops.",
    "Require wheelchair accessibility for all outings around Church Street.",
    "Outdoor cycling around Cubbon Park this Saturday morning, keeping it under ₹500.",
    "Also add linocut printmaking to my interests, keep my area as Koramangala."
]


# Concurrency note: 50 Locust users with 1.0-3.0s think time generate an arrival
# rate of ~10-15 req/s and approximately 30 simultaneous in-flight requests.


class Phase6DInferenceUser(HttpUser):
    # Simulate realistic user think time between conversation turns (1.0 to 3.0 seconds)
    wait_time = between(1.0, 3.0)
    endpoint_path = "/v1/extract_preferences"

    @task(6)
    def test_standard_preference_extraction(self):
        """Simulates standard user preference statements (interests, locations, budgets)."""
        prompt = random.choice(USER_UTTERANCES[:4])
        self._send_extraction_request(prompt, tag="standard_extraction")

    @task(2)
    def test_preference_clearing_or_correction(self):
        """Simulates explicit preference clearings and location/budget corrections."""
        prompt = random.choice(USER_UTTERANCES[3:6])
        self._send_extraction_request(prompt, tag="correction_clearance")

    @task(2)
    def test_safety_boundary_extraction(self):
        """Simulates dietary and accessibility constraint statements."""
        prompt = random.choice(USER_UTTERANCES[6:8])
        self._send_extraction_request(prompt, tag="safety_boundary")

    def _send_extraction_request(self, text: str, tag: str):
        payload = {
            "prompt": f"User message: {text}",
            "operation_name": "memory_extraction",
            "temperature": 0.3,
            "max_tokens": 150
        }
        headers = {
            "Content-Type": "application/json",
            "X-Client-Trace-Id": f"locust_{time.time()}_{random.randint(1000, 9999)}"
        }

        start_time = time.time()
        with self.client.post(
            self.endpoint_path,
            json=payload,
            headers=headers,
            timeout=18.0,
            catch_response=True,
            name=f"extract_{tag}"
        ) as response:

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                try:
                    data = response.json()
                    model_text = data.get("text", "")
                    if "{" in model_text and "}" in model_text:
                        response.success()
                    else:
                        response.failure(f"Malformed structure: missing JSON brackets in '{model_text[:80]}'")
                except Exception as exc:
                    response.failure(f"JSON decode failed: {exc}")
            elif response.status_code in [429, 503]:
                response.failure(f"Queue/Concurrency rejection: HTTP {response.status_code}")
            elif response.status_code == 504:
                response.failure("Timeout (504): Request queued beyond server execution window")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:100]}")
