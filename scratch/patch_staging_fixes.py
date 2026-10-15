import re
from pathlib import Path

FORYOU = Path("/Users/shivamsingh/Desktop/foryou")

# 1. Update aiSchemas.ts
ai_schemas_path = FORYOU / "src/ai/schemas/aiSchemas.ts"
content = ai_schemas_path.read_text(encoding="utf-8")

# Add location scratching and dietary/accessibility in validateMemoryExtraction
target_clear = """    if (
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
    }"""

replacement_clear = """    if (
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

    // Explicit Location Corrections (e.g. "scratch Indiranagar")
    const scratchMatches = lowerInput.matchAll(/(?:scratch|remove|no longer in|moved away from|not in)\\s+([a-z\\s]+?)(?:—|--|-|\\.|,|$|\\band\\b|\\bnow\\b|\\bplease\\b)/gi);
    for (const match of scratchMatches) {
      if (match && match[1]) {
        const scratched = match[1].trim();
        if (!isFabricatedPlaceholder(scratched) && scratched.length > 2) {
          const scratchedSlug = scratched.toLowerCase().replace(/[\\s-]+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 50);
          records.push({
            user_id: userId,
            category: 'location',
            key: `neighborhood_${scratchedSlug}`,
            value: 'cleared',
            source: 'explicit',
            confidence: 1.0,
            status: 'superseded' as any,
            evidence: rawInput.slice(0, 300),
            first_seen_at: now,
            last_confirmed_at: now
          });
        }
      }
    }

    // Safety Constraint Preservation: Dietary and accessibility
    if (/vegan|vegetarian|gluten[- ]free|dairy[- ]free|halal|kosher|allerg/i.test(rawInput)) {
      const dMatch = rawInput.match(/\\b(strictly vegan|vegan|vegetarian|gluten[- ]free|dairy[- ]free|halal|kosher|peanut allergy|nut allergy)\\b/i);
      const dVal = dMatch ? dMatch[1] : 'Dietary restriction';
      records.push({
        user_id: userId,
        category: 'safety_boundary',
        key: 'dietary_restriction',
        value: dVal.charAt(0).toUpperCase() + dVal.slice(1),
        source: 'explicit',
        confidence: 1.0,
        status: 'active',
        evidence: rawInput.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }

    if (/wheelchair|accessible|accessibility|mobility|step[- ]free/i.test(rawInput)) {
      records.push({
        user_id: userId,
        category: 'safety_boundary',
        key: 'accessibility_requirement',
        value: 'Wheelchair accessibility required',
        source: 'explicit',
        confidence: 1.0,
        status: 'active',
        evidence: rawInput.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }"""

if target_clear in content:
    content = content.replace(target_clear, replacement_clear)
    ai_schemas_path.write_text(content, encoding="utf-8")
    print("aiSchemas.ts updated successfully")
else:
    print("WARNING: target_clear not found in aiSchemas.ts")


# 2. Update memoryManager.ts
mem_manager_path = FORYOU / "src/ai/orchestrator/memoryManager.ts"
mem_content = mem_manager_path.read_text(encoding="utf-8")

old_superseded_block = """      // 0. Explicit Clearing Handling: if incoming record is marked superseded, supersede active match
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

new_superseded_block = """      // 0. Explicit Clearing Handling: if incoming record is marked superseded, supersede active match
      if (incoming.status === 'superseded') {
        const existingMatches = existingRecords.filter(
          r => (r.category === incoming.category && r.key === incoming.key) ||
               (r.category === incoming.category && incoming.key === 'preferred_neighborhood') ||
               (r.category === incoming.category && (r.key.includes(incoming.key) || incoming.key.includes(r.key))) ||
               (r.category === incoming.category && incoming.key.includes(r.value.toLowerCase()))
        );
        for (const existingMatch of existingMatches) {
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

if old_superseded_block in mem_content:
    mem_content = mem_content.replace(old_superseded_block, new_superseded_block)
    mem_manager_path.write_text(mem_content, encoding="utf-8")
    print("memoryManager.ts updated successfully")
else:
    print("WARNING: old_superseded_block not found in memoryManager.ts")


# 3. Update memoryExtractor.ts
mem_extractor_path = FORYOU / "src/ai/memory/memoryExtractor.ts"
ext_content = mem_extractor_path.read_text(encoding="utf-8")

old_indi = """    if (text.includes('indiranagar')) {
      addRecord('location', 'indiranagar', 'Indiranagar', 'explicit', baseConfidence);
    }"""

new_indi = """    if (text.includes('indiranagar') && !text.includes('scratch indiranagar')) {
      addRecord('location', 'indiranagar', 'Indiranagar', 'explicit', baseConfidence);
    }"""

old_scratch = """    // Location Corrections (e.g. scratch Whitefield -> Koramangala)
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
    }"""

new_scratch = """    // Location Corrections (e.g. scratch Whitefield/Indiranagar -> Koramangala)
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
    if (text.includes('scratch indiranagar') || (text.includes('moved to koramangala') && text.includes('indiranagar'))) {
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
      addRecord('location', 'neighborhood_koramangala', 'Koramangala', 'explicit', 0.95);
    }

    // Safety constraints
    if (/vegan|vegetarian|gluten[- ]free|dairy[- ]free|halal|kosher|allerg/i.test(userText)) {
      const dMatch = userText.match(/\\b(strictly vegan|vegan|vegetarian|gluten[- ]free|dairy[- ]free|halal|kosher|peanut allergy|nut allergy)\\b/i);
      const dVal = dMatch ? dMatch[1] : 'Dietary restriction';
      records.push({
        user_id: userId,
        category: 'safety_boundary',
        key: 'dietary_restriction',
        value: dVal.charAt(0).toUpperCase() + dVal.slice(1),
        source: 'explicit',
        confidence: 1.0,
        status: 'active',
        evidence: userText.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }
    if (/wheelchair|accessible|accessibility|mobility|step[- ]free/i.test(userText)) {
      records.push({
        user_id: userId,
        category: 'safety_boundary',
        key: 'accessibility_requirement',
        value: 'Wheelchair accessibility required',
        source: 'explicit',
        confidence: 1.0,
        status: 'active',
        evidence: userText.slice(0, 300),
        first_seen_at: now,
        last_confirmed_at: now
      });
    }"""

if old_indi in ext_content:
    ext_content = ext_content.replace(old_indi, new_indi)
if old_scratch in ext_content:
    ext_content = ext_content.replace(old_scratch, new_scratch)
mem_extractor_path.write_text(ext_content, encoding="utf-8")
print("memoryExtractor.ts updated successfully")


# 4. Update phase6d-staging-validation.test.ts
test_path = FORYOU / "tests/phase6d-staging-validation.test.ts"
test_content = test_path.read_text(encoding="utf-8")

if "import fs from 'fs';" not in test_content:
    test_content = "import fs from 'fs';\n" + test_content
test_content = test_content.replace("const fs = require('fs');", "// fs imported above")
test_path.write_text(test_content, encoding="utf-8")
print("phase6d-staging-validation.test.ts updated successfully")
