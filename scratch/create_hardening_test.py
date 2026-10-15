import os
from pathlib import Path

test_code = """/**
 * Flowers For You — Phase 6D Preference Extraction Hardening Regression Test Suite
 *
 * Verifies all 9 regression requirements specified in Phase 6D Task 4:
 * 1. Normal interest and neighborhood extraction.
 * 2. Numeric INR budget extraction.
 * 3. Missing preferences do not create fabricated values.
 * 4. Correcting a saved budget from ₹800 to ₹1500.
 * 5. Adding a new interest without losing existing interests.
 * 6. Explicitly clearing a preference.
 * 7. Invalid JSON and invalid field types are rejected.
 * 8. Existing memory records are preserved when unrelated fields are omitted.
 * 9. The planner receives the updated, validated preferences.
 */

import { db } from "../src/backend/core/database";
import { AuthService } from "../src/backend/auth/authService";
import { MemoryExtractor } from "../src/ai/memory/memoryExtractor";
import { MemoryManager } from "../src/ai/orchestrator/memoryManager";
import { ContextBuilder } from "../src/ai/orchestrator/contextBuilder";
import { ExperiencePlanner } from "../src/ai/planner/experiencePlanner";
import { SchemaValidator, isFabricatedPlaceholder } from "../src/ai/schemas/aiSchemas";
import { CurrentIntent } from "../src/types";

async function runPreferenceHardeningTests() {
  console.log('\\n🌸 =======================================================');
  console.log('   FLOWERS FOR YOU — PHASE 6D PREFERENCE HARDENING TESTS');
  console.log('=======================================================🌸\\n');

  let passed = 0;
  let total = 0;

  function assert(condition: boolean, testName: string, detail?: string) {
    total++;
    if (condition) {
      passed++;
      console.log(`  ✓ Passed: [${total}] ${testName} ${detail ? "(" + detail + ")" : ""}`);
    } else {
      console.error(`  ✗ FAILED: [${total}] ${testName} ${detail ? "[" + detail + "]" : ""}`);
      process.exit(1);
    }
  }

  const testUserId = `test_pref_user_${Date.now()}`;
  const now = new Date().toISOString();

  // =========================================================================
  // TEST 1: Normal interest and neighborhood extraction
  // =========================================================================
  console.log('--- TEST 1: Normal interest and neighborhood extraction ---');
  const payload1 = {
    action: 'save_preferences',
    interests: ['specialty_coffee', 'artisanal_pottery'],
    preferred_neighborhoods: ['Indiranagar']
  };
  const input1 = 'Please save my profile: my interests are specialty coffee and artisanal pottery, preferred area is Indiranagar.';
  const validated1 = SchemaValidator.validateMemoryExtraction(payload1, testUserId, input1);

  assert(validated1.records.length === 3, 'Extracts 3 structured records (2 interests, 1 location)');
  const interests1 = validated1.records.filter(r => r.category === 'interest');
  const locations1 = validated1.records.filter(r => r.category === 'location');
  assert(interests1.some(r => r.value.toLowerCase().includes('specialty coffee')), 'Extracts specialty coffee interest');
  assert(interests1.some(r => r.value.toLowerCase().includes('pottery')), 'Extracts artisanal pottery interest');
  assert(locations1.some(r => r.value === 'Indiranagar' && r.key === 'neighborhood_indiranagar'), 'Extracts Indiranagar neighborhood location');
  assert(validated1.records.every(r => r.source === 'explicit' && r.confidence >= 0.9 && r.status === 'active'), 'Metadata conforms to canonical schema');

  // =========================================================================
  // TEST 2: Numeric INR budget extraction
  // =========================================================================
  console.log('\\n--- TEST 2: Numeric INR budget extraction ---');
  const payload2 = {
    action: 'save_preferences',
    interests: ['board_games'],
    preferred_neighborhoods: ['Bandra West'],
    budget_inr_max: 1500
  };
  const input2 = 'Update my spending limit: my maximum budget is ₹1500 per person in Bandra West for indie board games.';
  const validated2 = SchemaValidator.validateMemoryExtraction(payload2, testUserId, input2);

  const budgetRecord2 = validated2.records.find(r => r.category === 'budget');
  assert(budgetRecord2 !== undefined, 'Extracts budget record');
  assert(budgetRecord2?.key === 'budget_ceiling_inr', 'Uses canonical key budget_ceiling_inr');
  assert(budgetRecord2?.value === '1500', 'Extracts exact numeric budget 1500 as string');
  assert(budgetRecord2?.confidence === 0.95 && budgetRecord2?.status === 'active', 'Budget record is active with 0.95 confidence');

  // =========================================================================
  // TEST 3: Missing preferences do not create fabricated values
  // =========================================================================
  console.log('\\n--- TEST 3: Missing preferences do not create fabricated values ---');
  assert(isFabricatedPlaceholder('city_a') === true, 'Placeholder checker catches city_a');
  assert(isFabricatedPlaceholder('city_b') === true, 'Placeholder checker catches city_b');
  assert(isFabricatedPlaceholder('unknown') === true, 'Placeholder checker catches unknown');
  assert(isFabricatedPlaceholder('N/A') === true, 'Placeholder checker catches N/A');
  assert(isFabricatedPlaceholder('Koramangala') === false, 'Real neighborhood Koramangala is not rejected');

  const payload3WithHallucinations = {
    action: 'save_preferences',
    interests: ['modern_strategy_board_games'],
    preferred_neighborhoods: ['city_a', 'city_b', 'unknown'],
    budget_inr_max: 100
  };
  const input3 = "I really enjoy modern strategy board games, but I haven't decided on any location or budget yet.";
  const validated3 = SchemaValidator.validateMemoryExtraction(payload3WithHallucinations, testUserId, input3);

  const locs3 = validated3.records.filter(r => r.category === 'location');
  const budgets3 = validated3.records.filter(r => r.category === 'budget');
  assert(locs3.length === 0, 'Rejected all fabricated placeholder neighborhoods');
  assert(budgets3.length === 0, 'Rejected fabricated budget when user explicitly disclaimed budget');
  assert(validated3.records.length === 1 && validated3.records[0].value.toLowerCase().includes('board games'), 'Preserved legitimate stated interest');

  // =========================================================================
  // TEST 4: Correcting a saved budget from ₹800 to ₹1500
  // =========================================================================
  console.log('\\n--- TEST 4: Correcting a saved budget from ₹800 to ₹1500 ---');
  // 4a. Persist initial budget ₹800
  await MemoryManager.reconcileAndSave(testUserId, [
    {
      user_id: testUserId,
      category: 'budget',
      key: 'budget_ceiling_inr',
      value: '800',
      source: 'explicit',
      confidence: 0.95,
      status: 'active',
      evidence: 'my budget is 800',
      first_seen_at: now,
      last_confirmed_at: now
    }
  ]);
  const initialMem = await MemoryManager.getActiveMemorySummary(testUserId);
  assert(initialMem.budget_preferences.includes('800'), 'Initial budget ₹800 successfully persisted');

  // 4b. Reconcile correction to ₹1500
  const correctionPayload = {
    action: 'save_preferences',
    budget_inr_max: 1500
  };
  const correctionInput = 'Actually update my budget: please raise my maximum budget to ₹1500.';
  const validatedCorrection = SchemaValidator.validateMemoryExtraction(correctionPayload, testUserId, correctionInput);
  await MemoryManager.reconcileAndSave(testUserId, validatedCorrection.records);

  const updatedMem = await MemoryManager.getActiveMemorySummary(testUserId);
  assert(updatedMem.budget_preferences.includes('1500'), 'Active budget preference updated to 1500');
  assert(!updatedMem.budget_preferences.includes('800'), 'Old budget ₹800 is no longer active');

  // Verify that active budget reflects correction and old budget is superseded
  const rawDb = await db.getUserStructuredMemory(testUserId);
  const activeBudget = (rawDb.raw_records || []).find(r => r.category === 'budget' && r.status === 'active');
  assert(activeBudget !== undefined && activeBudget.value === '1500', 'New ₹1500 budget record is active');
  assert(!rawDb.raw_records.some(r => r.category === 'budget' && r.status === 'active' && r.value === '800'), 'Previous ₹800 budget is superseded and not active');

  // =========================================================================
  // TEST 5: Adding a new interest without losing existing interests
  // =========================================================================
  console.log('\\n--- TEST 5: Adding a new interest without losing existing interests ---');
  // Seed two existing interests
  await MemoryManager.reconcileAndSave(testUserId, [
    {
      user_id: testUserId,
      category: 'interest',
      key: 'outdoor_cycling',
      value: 'Outdoor Cycling',
      source: 'explicit',
      confidence: 0.95,
      status: 'active',
      evidence: 'I like cycling',
      first_seen_at: now,
      last_confirmed_at: now
    },
    {
      user_id: testUserId,
      category: 'interest',
      key: 'sourdough_baking',
      value: 'Sourdough Baking',
      source: 'explicit',
      confidence: 0.95,
      status: 'active',
      evidence: 'I like sourdough baking',
      first_seen_at: now,
      last_confirmed_at: now
    }
  ]);

  const memBeforeAdd = await MemoryManager.getActiveMemorySummary(testUserId);
  assert(memBeforeAdd.interests.length >= 2, 'User has initial interests cycling and sourdough');

  // User adds specialty coffee
  const addPayload = {
    action: 'save_preferences',
    interests: ['specialty_coffee_tasting'],
    preferred_neighborhoods: ['Hauz Khas Village'],
    budget_inr_max: 800
  };
  const addInput = 'Also add specialty coffee tasting to my interests, keep it around ₹800 max in Hauz Khas Village.';
  const validatedAdd = SchemaValidator.validateMemoryExtraction(addPayload, testUserId, addInput);
  await MemoryManager.reconcileAndSave(testUserId, validatedAdd.records);

  const memAfterAdd = await MemoryManager.getActiveMemorySummary(testUserId);
  assert(memAfterAdd.interests.some(i => i.toLowerCase().includes('cycling')), 'Preserved Outdoor Cycling');
  assert(memAfterAdd.interests.some(i => i.toLowerCase().includes('sourdough')), 'Preserved Sourdough Baking');
  assert(memAfterAdd.interests.some(i => i.toLowerCase().includes('coffee')), 'Added Specialty Coffee Tasting');
  assert(memAfterAdd.interests.length >= 3, 'All 3 interests are active simultaneously');

  // =========================================================================
  // TEST 6: Explicitly clearing a preference
  // =========================================================================
  console.log('\\n--- TEST 6: Explicitly clearing a preference ---');
  // Clear budget
  const clearInput = "Please clear my budget limit, I don't want any budget ceiling.";
  const clearValidated = SchemaValidator.validateMemoryExtraction({}, testUserId, clearInput);
  assert(clearValidated.records.some(r => r.category === 'budget' && r.status === 'superseded'), 'Extraction produces superseded budget record for clear request');

  await MemoryManager.reconcileAndSave(testUserId, clearValidated.records);
  const memAfterClear = await MemoryManager.getActiveMemorySummary(testUserId);
  assert(memAfterClear.budget_preferences.length === 0, 'Active budget preferences are now empty after clearing');

  // Verify other preferences (interests) remain active
  assert(memAfterClear.interests.length >= 3, 'Interests remain completely intact after clearing budget');

  // =========================================================================
  // TEST 7: Invalid JSON and invalid field types are rejected
  // =========================================================================
  console.log('\\n--- TEST 7: Invalid JSON and invalid field types are rejected ---');
  let malformedCaught = false;
  try {
    SchemaValidator.validateMemoryExtraction('not valid json' as any, testUserId, 'test');
  } catch (e) {
    malformedCaught = true;
  }
  assert(malformedCaught, 'Throws error on non-object memory payload');

  // Invalid types inside payload
  const invalidTypesPayload = {
    action: 'save_preferences',
    interests: 12345, // invalid type
    preferred_neighborhoods: 'Indiranagar', // string instead of array
    budget_inr_max: 'expensive' // non-numeric string
  };
  const validatedInvalid = SchemaValidator.validateMemoryExtraction(invalidTypesPayload, testUserId, 'test');
  assert(validatedInvalid.records.length === 0, 'Safely rejects invalid non-array and non-numeric field types without crashing');

  // =========================================================================
  // TEST 8: Existing memory records are preserved when unrelated fields are omitted
  // =========================================================================
  console.log('\\n--- TEST 8: Existing memory records are preserved when unrelated fields are omitted ---');
  const isolateUser8 = `user_isolate_${Date.now()}`;
  await MemoryManager.reconcileAndSave(isolateUser8, [
    {
      user_id: isolateUser8,
      category: 'interest',
      key: 'linocut_printmaking',
      value: 'Linocut Printmaking',
      source: 'explicit',
      confidence: 0.95,
      status: 'active',
      evidence: 'I do linocut',
      first_seen_at: now,
      last_confirmed_at: now
    },
    {
      user_id: isolateUser8,
      category: 'budget',
      key: 'budget_ceiling_inr',
      value: '2000',
      source: 'explicit',
      confidence: 0.95,
      status: 'active',
      evidence: 'budget 2000',
      first_seen_at: now,
      last_confirmed_at: now
    },
    {
      user_id: isolateUser8,
      category: 'location',
      key: 'neighborhood_indiranagar',
      value: 'Indiranagar',
      source: 'explicit',
      confidence: 0.95,
      status: 'active',
      evidence: 'area Indiranagar',
      first_seen_at: now,
      last_confirmed_at: now
    }
  ]);

  // Update ONLY location
  const partialPayload = {
    action: 'save_preferences',
    preferred_neighborhoods: ['Koramangala']
  };
  const partialInput = 'Update my area to Koramangala.';
  const partialValidated = SchemaValidator.validateMemoryExtraction(partialPayload, isolateUser8, partialInput);
  await MemoryManager.reconcileAndSave(isolateUser8, partialValidated.records);

  const mem8 = await MemoryManager.getActiveMemorySummary(isolateUser8);
  assert(mem8.interests.includes('Linocut Printmaking'), 'Unrelated interest Linocut Printmaking preserved');
  assert(mem8.budget_preferences.includes('2000'), 'Unrelated budget 2000 preserved');
  assert(mem8.location_preferences.includes('Koramangala'), 'Updated location Koramangala active');

  // =========================================================================
  // TEST 9: The planner receives the updated, validated preferences
  // =========================================================================
  console.log('\\n--- TEST 9: The planner receives the updated, validated preferences ---');
  const activeMem = await MemoryManager.getActiveMemorySummary(isolateUser8);

  const testIntent: CurrentIntent = {
    raw_request: 'Plan a craft workshop for Saturday afternoon',
    activity: ['craft workshop', 'printmaking'],
    time: 'Saturday afternoon',
    group_size: 2,
    constraints: [],
    source: 'explicit'
  };

  const planningContext = ContextBuilder.buildPlanningContext(
    isolateUser8,
    activeMem,
    testIntent,
    ['User: I like linocut printmaking', 'User: Update my area to Koramangala']
  );

  assert(planningContext.relevantMemory.interests.some(i => i.toLowerCase().includes('linocut')), 'Planning context includes active interest');
  assert(planningContext.relevantMemory.location_preferences.includes('Koramangala'), 'Planning context includes updated location');
  assert(planningContext.relevantMemory.budget_preferences.includes('2000'), 'Planning context includes preserved budget 2000');

  const planResult = await ExperiencePlanner.generatePersonalizedPlan(planningContext);
  assert(planResult.plan !== undefined, 'Planner successfully generated proposed plan');
  assert(planResult.plan.title !== undefined && planResult.plan.title.length > 0, 'Plan has descriptive title');

  console.log('\\n=======================================================');
  console.log(`🌸 FINAL SCORE: ${passed}/${total} TESTS PASSED (100%)`);
  console.log('=======================================================\\n');
}

runPreferenceHardeningTests().catch(err => {
  console.error('Test execution failed:', err);
  process.exit(1);
});
"""

target = Path("/Users/shivamsingh/Desktop/foryou/tests/preference-hardening.test.ts")
target.write_text(test_code, encoding="utf-8")
print(f"Written {len(test_code)} chars to {target}")
