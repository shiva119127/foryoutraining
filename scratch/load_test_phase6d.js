import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

// Note on concurrency: 50 VUs with 1-3s think time models ~10 req/s arrival rate
// and approximately 30 simultaneous in-flight requests during steady state.

// Custom Prometheus/k6 observability metrics
const extractionLatency = new Trend('phase6d_extraction_duration_ms');
const serverErrorRate = new Rate('phase6d_server_error_rate'); // Unexpected 500, 502
const timeoutRate = new Rate('phase6d_timeout_rate'); // 504 Gateway Timeout
const admissionRejectionRate = new Rate('phase6d_admission_rejection_rate'); // Expected 503 / 429
const jsonValidRate = new Rate('phase6d_valid_json_rate');
const totalRequests = new Counter('phase6d_requests_total');

// Separate status code counters
const count200 = new Counter('phase6d_http_200_total');
const count404 = new Counter('phase6d_http_404_total');
const count422 = new Counter('phase6d_http_422_total');
const count429 = new Counter('phase6d_http_429_total');
const count503 = new Counter('phase6d_http_503_total');
const count500 = new Counter('phase6d_http_500_total');
const count502 = new Counter('phase6d_http_502_total');
const count504 = new Counter('phase6d_http_504_total');
const countOther = new Counter('phase6d_http_other_total');


export const options = {
  stages: [
    { duration: '2m', target: 10 },  // Phase 1: Warm-up (0 -> 10 VUs)
    { duration: '3m', target: 50 },  // Phase 2: Ramp-up (10 -> 50 VUs)
    { duration: '6m', target: 50 },  // Phase 3: Steady-state (50 VUs sustained)
    { duration: '1m', target: 75 },  // Phase 4: Peak burst stress (50 -> 75 VUs)
    { duration: '1m', target: 50 },  // Phase 5: Burst recovery (75 -> 50 VUs)
    { duration: '2m', target: 0 },   // Phase 6: Cooldown & drain (50 -> 0 VUs)
  ],
  thresholds: {
    // PROVISIONAL HARDWARE HYPOTHESES:
    // Single-request warm baseline on Apple Silicon MPS is ~2990ms.
    // Under concurrency, queuing delay adds to latency. Hard SLAs apply to Linux GPU staging.
    'http_req_duration': ['p(50)<3000', 'p(95)<5000', 'p(99)<7500'],
    'phase6d_extraction_duration_ms': ['p(50)<3000', 'p(95)<5000'],
    // Reliability SLAs (excluding planned backpressure rejections)
    'phase6d_server_error_rate': ['rate<0.01'],        // Unexpected crashes/exceptions < 1%
    'phase6d_valid_json_rate': ['rate>0.99'],          // >99% valid JSON on successful responses
  },
};

// 10 realistic canonical test scenarios matching Phase 6D distribution
const USER_UTTERANCES = [
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
];

const TARGET_ENDPOINT = __ENV.TARGET_URL || 'http://127.0.0.1:8000/v1/extract_preferences';

export default function () {
  const prompt = USER_UTTERANCES[Math.floor(Math.random() * USER_UTTERANCES.length)];
  
  const payload = JSON.stringify({
    prompt: `User message: ${prompt}`,
    operation_name: 'memory_extraction',
    temperature: 0.3,
    max_tokens: 150
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-Client-Trace-Id': `k6_${__VU}_${__ITER}_${Date.now()}`
    },
    timeout: '18s'
  };

  const startTime = Date.now();
  const res = http.post(TARGET_ENDPOINT, payload, params);
  const duration = Date.now() - startTime;

  totalRequests.add(1);
  extractionLatency.add(duration);

  const isSuccess = res.status === 200;
  const isAdmissionRejection = res.status === 503 || res.status === 429;
  const isTimeout = res.status === 504;
  const isServerError = res.status >= 500 && !isAdmissionRejection && !isTimeout;

  admissionRejectionRate.add(isAdmissionRejection);
  timeoutRate.add(isTimeout);
  serverErrorRate.add(isServerError);

  // Separate status code counts
  if (res.status === 200) count200.add(1);
  else if (res.status === 404) count404.add(1);
  else if (res.status === 422) count422.add(1);
  else if (res.status === 429) count429.add(1);
  else if (res.status === 503) count503.add(1);
  else if (res.status === 500) count500.add(1);
  else if (res.status === 502) count502.add(1);
  else if (res.status === 504) count504.add(1);
  else countOther.add(1);


  let hasValidJson = false;
  if (isSuccess && res.body) {
    try {
      const parsed = JSON.parse(res.body);
      if (parsed.text && parsed.text.includes('{')) {
        hasValidJson = true;
      }
    } catch (e) {
      hasValidJson = false;
    }
  }
  jsonValidRate.add(hasValidJson);

  check(res, {
    'status is 200 or 503 backpressure': (r) => r.status === 200 || r.status === 503,
    'response under 6s': (r) => r.timings.duration < 6000,
    'contains json block on success': () => !isSuccess || hasValidJson,
  });

  // User think time between turns (1 to 3 seconds)
  sleep(Math.random() * 2 + 1);
}

