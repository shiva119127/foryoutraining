import re
from pathlib import Path

FORYOU = Path("/Users/shivamsingh/Desktop/foryou")

# 1. Update memoryManager.ts
mem_manager_path = FORYOU / "src/ai/orchestrator/memoryManager.ts"
mem_content = mem_manager_path.read_text(encoding="utf-8")

old_dup_find = """      // 1. Check for exact duplicate / idempotent re-observation
      const existingMatch = existingRecords.find(
        r => r.category === incoming.category && r.key === incoming.key
      );

      if (existingMatch) {
        // Idempotency: Same value? Update confidence, confirmation timestamp, and keep active
        if (existingMatch.value.toLowerCase() === incoming.value.toLowerCase()) {
          reconciledToSave.push({
            user_id: userId,
            category: incoming.category,
            key: incoming.key,
            value: incoming.value,
            source: incoming.source === 'user_confirmed' ? 'user_confirmed' : existingMatch.source,
            confidence: Math.min(1.0, Math.max(existingMatch.confidence, incoming.confidence)),
            status: 'active',
            evidence: incoming.evidence || existingMatch.evidence,
            first_seen_at: existingMatch.first_seen_at || existingMatch.created_at || now,
            last_confirmed_at: now
          });
          continue;
        }"""

new_dup_find = """      // 1. Check for exact duplicate / idempotent re-observation
      const existingMatch = existingRecords.find(
        r => r.category === incoming.category && (
          r.key === incoming.key ||
          (r.category === 'interest' && (
            r.key.includes(incoming.key) ||
            incoming.key.includes(r.key) ||
            r.value.toLowerCase().includes(incoming.value.toLowerCase()) ||
            incoming.value.toLowerCase().includes(r.value.toLowerCase())
          ))
        )
      );

      if (existingMatch) {
        // Idempotency: Same or matching value? Update confidence, confirmation timestamp, and keep active
        const isSameOrMatching =
          existingMatch.value.toLowerCase() === incoming.value.toLowerCase() ||
          (incoming.category === 'interest' && (
            existingMatch.value.toLowerCase().includes(incoming.value.toLowerCase()) ||
            incoming.value.toLowerCase().includes(existingMatch.value.toLowerCase())
          ));

        if (isSameOrMatching) {
          reconciledToSave.push({
            user_id: userId,
            category: incoming.category,
            key: existingMatch.key,
            value: existingMatch.value,
            source: incoming.source === 'user_confirmed' ? 'user_confirmed' : existingMatch.source,
            confidence: Math.min(1.0, Math.max(existingMatch.confidence, incoming.confidence)),
            status: 'active',
            evidence: incoming.evidence || existingMatch.evidence,
            first_seen_at: existingMatch.first_seen_at || existingMatch.created_at || now,
            last_confirmed_at: now
          });
          continue;
        }"""

if old_dup_find in mem_content:
    mem_content = mem_content.replace(old_dup_find, new_dup_find)
    mem_manager_path.write_text(mem_content, encoding="utf-8")
    print("memoryManager.ts updated successfully")
else:
    print("WARNING: old_dup_find not found in memoryManager.ts")


# 2. Update 3e in phase6d-staging-validation.test.ts
test_path = FORYOU / "tests/phase6d-staging-validation.test.ts"
test_content = test_path.read_text(encoding="utf-8")

old_test_3e = """  // 3e. Duplicate extraction requests and idempotency
  console.log('  Testing 3e: Duplicate request idempotency...');
  const dupInput = "I like pottery and specialty coffee.";
  await AIOrchestrator.handleOnboardingTurn(user3, dupInput);
  await AIOrchestrator.handleOnboardingTurn(user3, dupInput);
  const memDup = await MemoryManager.getActiveMemorySummary(user3);
  const potteryCount = memDup.interests.filter(i => i.toLowerCase().includes('pottery')).length;
  assert(potteryCount === 1, 'Duplicate utterances do not create duplicate memory entries (idempotent)');"""

new_test_3e = """  // 3e. Duplicate extraction requests and idempotency
  console.log('  Testing 3e: Duplicate request idempotency...');
  const dupUser = `stg_user_dup_${Date.now()}`;
  const dupInput = "I like pottery and specialty coffee.";
  await AIOrchestrator.handleOnboardingTurn(dupUser, dupInput);
  const memDup1 = await MemoryManager.getActiveMemorySummary(dupUser);
  await AIOrchestrator.handleOnboardingTurn(dupUser, dupInput);
  const memDup2 = await MemoryManager.getActiveMemorySummary(dupUser);
  const potteryCount = memDup2.interests.filter(i => i.toLowerCase().includes('pottery')).length;
  assert(potteryCount === 1, 'Duplicate utterances do not create duplicate memory entries (idempotent)');
  assert(memDup1.interests.length === memDup2.interests.length, 'Memory count remains identical after duplicate retry');"""

if old_test_3e in test_content:
    test_content = test_content.replace(old_test_3e, new_test_3e)
    test_path.write_text(test_content, encoding="utf-8")
    print("phase6d-staging-validation.test.ts updated successfully")
else:
    print("WARNING: old_test_3e not found in phase6d-staging-validation.test.ts")
