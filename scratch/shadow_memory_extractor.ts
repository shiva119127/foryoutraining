/**
 * Flowers For You — Phase 6D Shadow Memory Extraction Module
 * 
 * Implements non-blocking, asynchronous shadow inference for Phase 6D.
 * Guarantees:
 * 1. Zero customer request latency impact (fire-and-forget background execution).
 * 2. Zero memory mutations (shadow extractions are strictly read-only and never saved to user memory).
 * 3. Zero side effects on failover or timeouts (all shadow errors are caught and logged).
 * 4. Structured metric comparison between primary cloud output and Phase 6D shadow output.
 * 5. Data minimization: No raw prompt text, no raw user utterances, and no plaintext preference values persisted.
 * 6. Pseudonymization: User IDs are hashed via SHA-256 (pseudonymized, not guaranteed anonymized).
 */

import { StructuredMemoryRecord, ExecutionMetadata } from '@/types';
import { SchemaValidator, isFabricatedPlaceholder } from './schemas/aiSchemas';
import crypto from 'crypto';

export interface ShadowComparisonRecord {
  requestId: string;
  timestamp: string;
  /** Pseudonymized identifier via SHA-256 (pseudonymized, not anonymous) */
  userIdPseudonym: string;
  rawInputLength: number;
  primaryProvider: string;
  primaryLatencyMs: number;
  primaryInterestCount: number;
  primaryLocationCount: number;
  primaryHasBudget: boolean;
  shadowProvider: string;
  shadowLatencyMs: number;
  shadowInterestCount: number;
  shadowLocationCount: number;
  shadowHasBudget: boolean;
  matchStatus: 'exact_match' | 'partial_match' | 'disagreement' | 'shadow_failure';
  discrepancies: string[];
  hallucinationsDetected: string[];
  errorSnippet?: string;
}

export class ShadowMemoryExtractor {
  private static endpoint = process.env.PHASE6D_SHADOW_ENDPOINT || 'http://127.0.0.1:8000';
  private static enabled = process.env.SHADOW_PHASE6D_ENABLED === 'true';
  private static timeoutMs = 8000;

  /**
   * Dispatches shadow extraction asynchronously in the background.
   * NEVER awaits in the main request pipeline!
   */
  static triggerShadowEvaluation(
    userText: string,
    userId: string,
    primaryRecords: Omit<StructuredMemoryRecord, 'id' | 'created_at' | 'updated_at'>[],
    primaryMetadata: ExecutionMetadata,
    requestId: string
  ): void {
    if (!this.enabled) return;

    // Fire and forget via setImmediate to free up Node event loop
    setImmediate(() => {
      this.executeShadowRun(userText, userId, primaryRecords, primaryMetadata, requestId)
        .catch(err => {
          console.warn(`[SHADOW EXTRACTION ERROR] Request ${requestId} failed silently: ${err.message}`);
        });
    });
  }

  private static async executeShadowRun(
    userText: string,
    userId: string,
    primaryRecords: Omit<StructuredMemoryRecord, 'id' | 'created_at' | 'updated_at'>[],
    primaryMetadata: ExecutionMetadata,
    requestId: string
  ): Promise<void> {
    const t0 = Date.now();
    let shadowRawText = '';
    let shadowLatencyMs = 0;
    let shadowRecords: Omit<StructuredMemoryRecord, 'id' | 'created_at' | 'updated_at'>[] = [];
    const discrepancies: string[] = [];
    const hallucinations: string[] = [];
    const pseudonym = crypto.createHash('sha256').update(userId).digest('hex').slice(0, 16);

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const res = await fetch(`${this.endpoint}/v1/extract_preferences`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Shadow-Request': 'true',
          'X-Request-Id': requestId
        },
        body: JSON.stringify({
          prompt: `User message: ${userText}`,
          operation_name: 'memory_extraction',
          temperature: 0.3,
          max_tokens: 150
        }),
        signal: controller.signal
      });
      clearTimeout(timer);

      shadowLatencyMs = Date.now() - t0;

      if (!res.ok) {
        throw new Error(`Shadow HTTP ${res.status}`);
      }

      const json = await res.json();
      shadowRawText = json.text || '';

      // Parse JSON from code block
      let parsedPayload: any = null;
      const jsonMatch = shadowRawText.match(/```(?:json)?\s*([\s\S]*?)\s*```/) || shadowRawText.match(/(\{[\s\S]*\})/);
      if (jsonMatch) {
        parsedPayload = JSON.parse(jsonMatch[1]);
      } else {
        throw new Error('Shadow model response omitted valid JSON block');
      }

      // Read-only schema validation
      const validated = SchemaValidator.validateMemoryExtraction(parsedPayload, userId, userText);
      shadowRecords = validated.records;

    } catch (err: any) {
      discrepancies.push(`Shadow failure: ${err.message}`);
      const comparison: ShadowComparisonRecord = {
        requestId,
        timestamp: new Date().toISOString(),
        userIdPseudonym: pseudonym,
        rawInputLength: userText.length,
        primaryProvider: primaryMetadata.provider,
        primaryLatencyMs: primaryMetadata.latency_ms,
        primaryInterestCount: primaryRecords.filter(r => r.category === 'interest').length,
        primaryLocationCount: primaryRecords.filter(r => r.category === 'location').length,
        primaryHasBudget: primaryRecords.some(r => r.category === 'budget'),
        shadowProvider: 'phase6d-shadow',
        shadowLatencyMs: shadowLatencyMs || (Date.now() - t0),
        shadowInterestCount: 0,
        shadowLocationCount: 0,
        shadowHasBudget: false,
        matchStatus: 'shadow_failure',
        discrepancies,
        hallucinationsDetected: [],
        errorSnippet: String(err.message).slice(0, 100)
      };
      await this.persistShadowLog(comparison);
      return;
    }

    // Compare Primary vs Shadow Extractions
    const primInterests = primaryRecords.filter(r => r.category === 'interest').map(r => r.value.toLowerCase());
    const shadInterests = shadowRecords.filter(r => r.category === 'interest').map(r => r.value.toLowerCase());

    const primLocs = primaryRecords.filter(r => r.category === 'location').map(r => r.value.toLowerCase());
    const shadLocs = shadowRecords.filter(r => r.category === 'location').map(r => r.value.toLowerCase());

    const primBudget = primaryRecords.find(r => r.category === 'budget')?.value || null;
    const shadBudget = shadowRecords.find(r => r.category === 'budget')?.value || null;

    // Check for placeholder emergence in shadow output
    for (const loc of shadLocs) {
      if (isFabricatedPlaceholder(loc)) {
        hallucinations.push(`Placeholder location: ${loc}`);
      }
    }

    // Evaluate Diff
    let matchStatus: 'exact_match' | 'partial_match' | 'disagreement' = 'exact_match';

    const interestMismatch = primInterests.some(pi => !shadInterests.some(si => si.includes(pi) || pi.includes(si))) ||
                             shadInterests.some(si => !primInterests.some(pi => pi.includes(si) || si.includes(pi)));
    const locMismatch = primLocs.sort().join(',') !== shadLocs.sort().join(',');
    const budgetMismatch = primBudget !== shadBudget;

    if (interestMismatch || locMismatch || budgetMismatch || hallucinations.length > 0) {
      matchStatus = (primInterests.length > 0 && shadInterests.length > 0) ? 'partial_match' : 'disagreement';
      if (interestMismatch) discrepancies.push(`Interest count/content divergence`);
      if (locMismatch) discrepancies.push(`Location divergence`);
      if (budgetMismatch) discrepancies.push(`Budget divergence`);
    }

    const comparison: ShadowComparisonRecord = {
      requestId,
      timestamp: new Date().toISOString(),
      userIdPseudonym: pseudonym,
      rawInputLength: userText.length,
      primaryProvider: primaryMetadata.provider,
      primaryLatencyMs: primaryMetadata.latency_ms,
      primaryInterestCount: primInterests.length,
      primaryLocationCount: primLocs.length,
      primaryHasBudget: primBudget !== null,
      shadowProvider: 'phase6d-shadow',
      shadowLatencyMs,
      shadowInterestCount: shadInterests.length,
      shadowLocationCount: shadLocs.length,
      shadowHasBudget: shadBudget !== null,
      matchStatus,
      discrepancies,
      hallucinationsDetected: hallucinations
    };

    // STRICT SAFETY GUARANTEE: Never call MemoryManager.reconcileAndSave() on shadow records!
    await this.persistShadowLog(comparison);
  }

  private static async persistShadowLog(log: ShadowComparisonRecord): Promise<void> {
    try {
      // Asynchronous, non-blocking file append with restricted permissions (0600)
      const fs = await import('fs');
      const path = await import('path');
      const logDir = path.join(process.cwd(), 'reports/shadow_logs');
      await fs.promises.mkdir(logDir, { recursive: true });
      const logFile = path.join(logDir, `shadow_eval_${new Date().toISOString().slice(0, 10)}.jsonl`);
      await fs.promises.appendFile(logFile, JSON.stringify(log) + '\n', { encoding: 'utf-8', mode: 0o600 });
    } catch (err: any) {
      // Logging failure must NEVER interrupt application response or bubble up
      console.warn(`[SHADOW AUDIT LOG WARNING] Failed to persist shadow record asynchronously: ${err.message}`);
    }
  }
}

