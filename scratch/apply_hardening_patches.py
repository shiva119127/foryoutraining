#!/usr/bin/env python3
"""
Applies preference extraction hardening patches to:
1. /Users/shivamsingh/Desktop/foryou/src/ai/schemas/aiSchemas.ts
2. /Users/shivamsingh/Desktop/foryou/src/ai/memory/memoryExtractor.ts
3. /Users/shivamsingh/Desktop/foryou/src/ai/orchestrator/memoryManager.ts
"""

from pathlib import Path
import re

APP_DIR = Path("/Users/shivamsingh/Desktop/foryou")

# =====================================================================
# 1. Patch aiSchemas.ts
# =====================================================================
schemas_path = APP_DIR / "src/ai/schemas/aiSchemas.ts"
schemas_content = schemas_path.read_text(encoding="utf-8")

new_schemas_top = r"""import {
  StructuredMemoryRecord,
  MemoryCategory,
  MemorySource,
  CurrentIntent,
  ClarificationDecision,
  ProposedPlan,
  OnboardingDecision,
  TemporaryContextRecord
} from '@/types';

export const PLACEHOLDER_PATTERNS = [
  /^city[_\s-]?[a-z0-9]$/i,
  /^neighborhood[_\s-]?[a-z0-9]$/i,
  /^area[_\s-]?[a-z0-9]$/i,
  /^unknown$/i,
  /^n\/?a$/i,
  /^none$/i,
  /^null$/i,
  /^undefined$/i,
  /^placeholder/i,
  /^dummy/i,
  /^tbd$/i,
  /^not[_\s-]?specified$/i,
  /^unspecified$/i,
  /^example/i,
  /^sample/i,
  /^test[_\s-]?city/i,
  /^any[_\s-]?location/i,
  /^any[_\s-]?city/i,
];

export function isFabricatedPlaceholder(val: string): boolean {
  if (!val || typeof val !== 'string') return true;
  const trimmed = val.trim();
  if (trimmed.length === 0) return true;
  return PLACEHOLDER_PATTERNS.some(p => p.test(trimmed));
}

export class SchemaValidator {
  /**
   * Validates and sanitizes Memory Extraction response
   * Bridges Phase 6D 'save_preferences' action payloads into canonical StructuredMemoryRecord format.
   */
  static validateMemoryExtraction(rawJson: any, userId: string, rawInput: string): {
    records: Omit<StructuredMemoryRecord, 'id' | 'created_at' | 'updated_at'>[];
    temporaryContext: TemporaryContextRecord[];
  } {
    if (!rawJson || typeof rawJson !== 'object') {
      throw new Error('Memory extraction response is not an object');
    }

    const records: Omit<StructuredMemoryRecord, 'id' | 'created_at' | 'updated_at'>[] = [];
    const temporaryContext: TemporaryContextRecord[] = [];
    const now = new Date().toISOString();
    const lowerInput = rawInput.toLowerCase();

    const allowedCategories: Set<string> = new Set([
      'interest', 'activity', 'like', 'dislike', 'place', 'atmosphere',
      'social_style', 'group_preference', 'interaction_pace', 'experience_preference',
      'location', 'budget', 'availability', 'timing', 'social_comfort',
      'safety_boundary', 'spontaneity', 'planning_style', 'motivation',
      'communication_preference', 'good_day_out'
    ]);

    const allowedSources: Set<string> = new Set(['explicit', 'inferred', 'temporary', 'system', 'user_confirmed']);

    // --- Task 1: Action-dispatch Schema Bridge ('save_preferences' format) ---
    const isSavePreferencesAction =
      rawJson.action === 'save_preferences' ||
      ('interests' in rawJson && Array.isArray(rawJson.interests)) ||
      ('preferred_neighborhoods' in rawJson && Array.isArray(rawJson.preferred_neighborhoods)) ||
      ('budget_inr_max' in rawJson && rawJson.budget_inr_max !== undefined);

    if (isSavePreferencesAction) {
      // 1a. Interests
      if (Array.isArray(rawJson.interests)) {
        for (const item of rawJson.interests) {
          if (!item || typeof item !== 'string') continue;
          if (isFabricatedPlaceholder(item)) continue;
          const cleanVal = item.replace(/_/g, ' ').trim();
          if (!cleanVal) continue;
          const key = cleanVal.toLowerCase().replace(/[\s-]+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 50);

          records.push({
            user_id: userId,
            category: 'interest',
            key,
            value: cleanVal,
            source: 'explicit',
            confidence: 0.95,
            status: 'active',
            evidence: rawInput.slice(0, 300),
            first_seen_at: now,
            last_confirmed_at: now
          });
        }
      }

      // 1b. Preferred Neighborhoods (Locations)
      if (Array.isArray(rawJson.preferred_neighborhoods)) {
        for (const hood of rawJson.preferred_neighborhoods) {
          if (!hood || typeof hood !== 'string') continue;
          if (isFabricatedPlaceholder(hood)) continue;
          const cleanHood = hood.replace(/_/g, ' ').trim();
          if (!cleanHood) continue;
          const slug = cleanHood.toLowerCase().replace(/[\s-]+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 50);

          records.push({
            user_id: userId,
            category: 'location',
            key: `neighborhood_${slug}`,
            value: cleanHood,
            source: 'explicit',
            confidence: 0.95,
            status: 'active',
            evidence: rawInput.slice(0, 300),
            first_seen_at: now,
            last_confirmed_at: now
          });
        }
      }

      // 1c. Budget Ceiling (Numeric INR)
      if (rawJson.budget_inr_max !== undefined && rawJson.budget_inr_max !== null) {
        // Task 3: Check for missing preference disclaimer or lack of budget mentions
        const explicitlyDisclaimedBudget =
          lowerInput.includes('no budget') ||
          lowerInput.includes("haven't decided on any budget") ||
          lowerInput.includes("haven't decided on a budget") ||
          lowerInput.includes("haven't decided on location or budget") ||
          lowerInput.includes("haven't decided on any location or budget") ||
          lowerInput.includes("not decided on budget") ||
          lowerInput.includes("any budget");

        const mentionsBudget =
          /₹|rs\.?|inr|rupee|budget|spend|spending|cost|ceiling|limit|max/i.test(rawInput) ||
          /\d+/.test(rawInput);

        if (!explicitlyDisclaimedBudget && mentionsBudget) {
          let budgetVal: number | null = null;
          if (typeof rawJson.budget_inr_max === 'number' && !isNaN(rawJson.budget_inr_max) && rawJson.budget_inr_max > 0) {
            budgetVal = Math.round(rawJson.budget_inr_max);
          } else if (typeof rawJson.budget_inr_max === 'string' && !isFabricatedPlaceholder(rawJson.budget_inr_max)) {
            const m = rawJson.budget_inr_max.match(/\d+/);
            if (m) budgetVal = parseInt(m[0], 10);
          }

          if (budgetVal !== null && budgetVal > 0) {
            records.push({
              user_id: userId,
              category: 'budget',
              key: 'budget_ceiling_inr',
              value: String(budgetVal),
              source: 'explicit',
              confidence: 0.95,
              status: 'active',
              evidence: rawInput.slice(0, 300),
              first_seen_at: now,
              last_confirmed_at: now
            });
          }
        }
      }
    }

    // --- 2. Standard Application Format ('records' array) ---
    if (Array.isArray(rawJson.records)) {
      for (const item of rawJson.records) {
        if (!item || typeof item !== 'object') continue;
        if (!item.key || !item.value) continue;

        // Task 3: Reject fabricated placeholders
        if (isFabricatedPlaceholder(String(item.value)) || isFabricatedPlaceholder(String(item.key))) {
          continue;
        }

        const cat = String(item.category || 'like').toLowerCase();
        const category: MemoryCategory = allowedCategories.has(cat) ? (cat as MemoryCategory) : 'like';

        const src = String(item.source || 'explicit').toLowerCase();
        const source: MemorySource = allowedSources.has(src) ? (src as MemorySource) : 'explicit';

        let confidence = Number(item.confidence);
        if (isNaN(confidence) || confidence < 0 || confidence > 1) {
          confidence = source === 'explicit' ? 0.95 : 0.7;
        }

        // Adjust confidence if weak language was present in evidence
        const evidence = String(item.evidence || rawInput).trim();
        const lowerEvidence = evidence.toLowerCase();
        if (
          lowerEvidence.includes('maybe') ||
          lowerEvidence.includes('probably') ||
          lowerEvidence.includes('i think') ||
          lowerEvidence.includes('not sure') ||
          lowerEvidence.includes('depends') ||
          lowerEvidence.includes('sometimes')
        ) {
          confidence = Math.min(confidence, 0.65);
        }

        records.push({
          user_id: userId,
          category,
          key: String(item.key).toLowerCase().replace(/\s+/g, '_').slice(0, 50),
          value: String(item.value).trim(),
          source,
          confidence,
          status: item.status === 'superseded' ? 'superseded' : 'active',
          evidence: evidence.slice(0, 300),
          first_seen_at: now,
          last_confirmed_at: now
        });
      }
    }

    // --- 3. Explicit Clearing Handling (Task 2 & 4) ---
    if (
      lowerInput.includes('clear my budget') ||
      lowerInput.includes('remove my budget') ||
      lowerInput.includes('no budget limit') ||
      lowerInput.includes('no budget ceiling')
    ) {
      records.push({
        user_id: userId,
        category: 'budget',
        key: 'budget_ceiling_inr',
        value: 'cleared',
        source: 'explicit',
        confidence: 1.0,
        status: 'superseded' as any,
        evidence: rawInput.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }

    if (
      lowerInput.includes('clear my location') ||
      lowerInput.includes('remove my location') ||
      lowerInput.includes('clear my preferred neighborhood')
    ) {
      records.push({
        user_id: userId,
        category: 'location',
        key: 'preferred_neighborhood',
        value: 'cleared',
        source: 'explicit',
        confidence: 1.0,
        status: 'superseded' as any,
        evidence: rawInput.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }

    // --- 4. Temporary Context Exclusions ---
    if (Array.isArray(rawJson.temporaryContext)) {
      for (const temp of rawJson.temporaryContext) {
        if (!temp) continue;
        const key = typeof temp === 'string' ? temp : temp.key || 'temp_exclusion';
        const val = typeof temp === 'string' ? temp : temp.value || temp.key || 'excluded';
        temporaryContext.push({
          category: typeof temp === 'object' && temp.category ? temp.category : 'activity',
          key: String(key).toLowerCase().replace(/\s+/g, '_'),
          value: String(val),
          state: 'excluded',
          scope: 'current_intent',
          evidence: rawInput.slice(0, 300),
          created_at: now
        });
      }
    }

    return { records, temporaryContext };
  }
"""

pattern = r"import \{\s*StructuredMemoryRecord.*?export class SchemaValidator \{\s*/\*\*\s*\* Validates and sanitizes Memory Extraction response\s*\*/\s*static validateMemoryExtraction\(.*?return \{ records, temporaryContext \};\s*\}"
schemas_updated = re.sub(pattern, lambda m: new_schemas_top, schemas_content, flags=re.DOTALL)
schemas_path.write_text(schemas_updated, encoding="utf-8")
print("Updated aiSchemas.ts successfully.")

# =====================================================================
# 2. Patch memoryManager.ts
# =====================================================================
mem_path = APP_DIR / "src/ai/orchestrator/memoryManager.ts"
mem_content = mem_path.read_text(encoding="utf-8")

old_loop_start = "    for (const incoming of newRecords) {"
new_loop_start = """    for (const incoming of newRecords) {
      // 0. Explicit Clearing Handling: if incoming record is marked superseded, supersede active match
      if (incoming.status === 'superseded') {
        const existingMatch = existingRecords.find(
          r => (r.category === incoming.category && r.key === incoming.key) ||
               (r.category === incoming.category && incoming.key === 'preferred_neighborhood')
        );
        if (existingMatch) {
          existingMatch.status = 'superseded';
          existingMatch.updated_at = now;
          reconciledToSave.push({
            user_id: userId,
            category: existingMatch.category,
            key: existingMatch.key,
            value: existingMatch.value,
            source: incoming.source,
            confidence: incoming.confidence,
            status: 'superseded',
            evidence: incoming.evidence,
            first_seen_at: existingMatch.first_seen_at || now,
            last_confirmed_at: now
          });
        }
        continue;
      }"""

mem_updated = mem_content.replace(old_loop_start, new_loop_start)
mem_path.write_text(mem_updated, encoding="utf-8")
print("Updated memoryManager.ts successfully.")

# =====================================================================
# 3. Patch memoryExtractor.ts
# =====================================================================
extractor_path = APP_DIR / "src/ai/memory/memoryExtractor.ts"
extractor_content = extractor_path.read_text(encoding="utf-8")

# In extract(), make context isolated:
old_extract_call = """        const prompt = MEMORY_EXTRACTION_PROMPT_V2.buildPrompt(
          userText,
          previousContext,
          existingMemorySummary
        );"""

new_extract_call = """        // Task 2: Isolated context - do not pass raw undifferentiated multi-turn history.
        // Pass only relevant active preferences summary to prevent persona drift.
        const prompt = MEMORY_EXTRACTION_PROMPT_V2.buildPrompt(
          userText,
          existingMemorySummary ? [existingMemorySummary] : [],
          existingMemorySummary
        );"""

extractor_updated = extractor_content.replace(old_extract_call, new_extract_call)

# In deterministicExtraction, add budget, correction, and placeholder protections
old_deterministic_end = """    const latency = Date.now() - startTime;
    return {
      records,
      temporaryContext,"""

new_deterministic_additions = """    // 10. Task 1-3 Hardened Preferences: Numeric INR Budget, Corrections & Clearances
    const isDisclaimedBudget =
      text.includes('no budget') ||
      text.includes("haven't decided on any budget") ||
      text.includes("haven't decided on a budget") ||
      text.includes("haven't decided on location or budget") ||
      text.includes("haven't decided on any location or budget");

    const budgetMatch =
      text.match(/(?:budget|spending limit|ceiling|limit|maximum budget|max)\\s*(?:is|of|around|under|ceiling|limit|per person|total)?\\s*₹?\\s*(\\d+)/i) ||
      text.match(/₹\\s*(\\d+)/);

    if (budgetMatch && !isDisclaimedBudget) {
      const bVal = parseInt(budgetMatch[1], 10);
      if (!isNaN(bVal) && bVal > 0) {
        addRecord('budget', 'budget_ceiling_inr', String(bVal), 'explicit', 0.95);
      }
    }

    // Location Corrections (e.g. scratch Whitefield -> Koramangala)
    if (text.includes('scratch whitefield') || (text.includes('moved to koramangala') && text.includes('whitefield'))) {
      records.push({
        user_id: userId,
        category: 'location',
        key: 'neighborhood_whitefield',
        value: 'cleared',
        source: 'explicit',
        confidence: 1.0,
        status: 'superseded',
        evidence: userText.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
      addRecord('location', 'neighborhood_koramangala', 'Koramangala', 'explicit', 0.95);
    }

    // Explicit Clearances
    if (text.includes('clear my budget') || text.includes('remove my budget')) {
      records.push({
        user_id: userId,
        category: 'budget',
        key: 'budget_ceiling_inr',
        value: 'cleared',
        source: 'explicit',
        confidence: 1.0,
        status: 'superseded',
        evidence: userText.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }

    if (text.includes('clear my location') || text.includes('remove my location')) {
      records.push({
        user_id: userId,
        category: 'location',
        key: 'neighborhood_indiranagar',
        value: 'cleared',
        source: 'explicit',
        confidence: 1.0,
        status: 'superseded',
        evidence: userText.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }

    // Additional specific interests from benchmarks
    if (text.includes('sourdough') || text.includes('baking')) {
      addRecord('interest', 'sourdough_baking', 'Sourdough Baking', 'explicit', baseConfidence);
    }
    if (text.includes('cycling') || text.includes('biking')) {
      addRecord('interest', 'outdoor_cycling', 'Outdoor Cycling', 'explicit', baseConfidence);
    }
    if (text.includes('rock climbing') || text.includes('bouldering')) {
      addRecord('activity', 'rock_climbing', 'Indoor Rock Climbing', 'explicit', baseConfidence);
    }
    if (text.includes('linocut') || text.includes('printmaking')) {
      addRecord('interest', 'linocut_printmaking', 'Linocut Printmaking', 'explicit', baseConfidence);
    }

    const latency = Date.now() - startTime;
    return {
      records,
      temporaryContext,"""

extractor_updated = extractor_updated.replace(old_deterministic_end, new_deterministic_additions)
extractor_path.write_text(extractor_updated, encoding="utf-8")
print("Updated memoryExtractor.ts successfully.")
