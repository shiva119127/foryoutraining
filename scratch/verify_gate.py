#!/usr/bin/env python3
"""
FlowersForYou - Phase 6D Safety Gate Verification Test
Tests both:
1. Configuration Gate: loadAndValidateConfig() throws when AI_ENV=production & STAGING_PHASE6D=true
2. Provider Resolution Gate: getActiveLlmClient() throws when AI_ENV=production & STAGING_PHASE6D=true
3. Isolation Verification: getActiveLlmClient() never returns Phase6D adapter in clean production
"""

import subprocess
import sys

# Subtest 1: Verify safety gates trip on forbidden staging flags in production
script_tripping_gates = """
async function testTrippingGates() {
  process.env.AI_ENV = "production";
  process.env.STAGING_PHASE6D = "true";

  // Test 1: Config schema gate
  try {
    const { loadAndValidateConfig } = await import("./src/config/environment.ts");
    loadAndValidateConfig();
    console.error("[FAIL] Config safety gate failed to trip!");
    process.exit(1);
  } catch (e) {
    if (e.message.includes("[CRITICAL SAFETY GATE]")) {
      console.log("[PASS] Config Safety Gate tripped correctly:", e.message);
    } else {
      console.error("[FAIL] Config threw unexpected error:", e.message);
      process.exit(1);
    }
  }

  // Test 2: Runtime Provider Resolution Gate
  try {
    const { getActiveLlmClient } = await import("./src/ai/provider/llmProvider.ts");
    getActiveLlmClient();
    console.error("[FAIL] Runtime provider safety gate failed to trip!");
    process.exit(1);
  } catch (e) {
    if (e.message.includes("[CRITICAL SAFETY GATE]")) {
      console.log("[PASS] Runtime Provider Safety Gate tripped correctly:", e.message);
    } else {
      console.error("[FAIL] Provider threw unexpected error:", e.message);
      process.exit(1);
    }
  }
  process.exit(0);
}
testTrippingGates().catch(e => { console.error(e); process.exit(2); });
"""

res1 = subprocess.run(
    ["node", "--import", "tsx", "-e", script_tripping_gates],
    cwd="/Users/shivamsingh/Desktop/foryou",
    capture_output=True,
    text=True
)

print(res1.stdout.strip())
if res1.returncode != 0:
    print("STDERR:\n", res1.stderr)
    sys.exit(res1.returncode)

# Subtest 2: Verify that clean production NEVER resolves to Phase 6D local adapter
script_clean_production = """
async function testCleanProduction() {
  delete process.env.STAGING_PHASE6D;
  delete process.env.USE_PHASE6D_ADAPTER;
  delete process.env.AI_PROVIDER;
  process.env.AI_ENV = "production";

  const { getActiveLlmClient } = await import("./src/ai/provider/llmProvider.ts");
  const client = getActiveLlmClient();

  if (client && client.providerName === "phase6d-adapter") {
    console.error("[FAIL] Clean production resolved to Phase 6D local adapter!");
    process.exit(1);
  }

  const resolved = client ? client.providerName : "null (no cloud API keys in test environment)";
  console.log("[PASS] Clean production resolved safely:", resolved);
  console.log("ALL SAFETY GATES PASSED CLEANLY (Zero Production Leakage).");
  process.exit(0);
}
testCleanProduction().catch(e => { console.error(e); process.exit(2); });
"""

res2 = subprocess.run(
    ["node", "--import", "tsx", "-e", script_clean_production],
    cwd="/Users/shivamsingh/Desktop/foryou",
    capture_output=True,
    text=True
)

print(res2.stdout.strip())
if res2.returncode != 0:
    print("STDERR:\n", res2.stderr)
    sys.exit(res2.returncode)

print("\nRETURN CODE: 0")
sys.exit(0)


