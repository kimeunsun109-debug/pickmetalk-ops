import {
  LORA_GENERATION_METHOD,
  LORA_IDENTITY_THRESHOLDS,
  methodIsForbidden,
} from '../../config/character-lora.config.js';

export type IdentityEngine = 'arcface' | 'proxy' | 'none';

export type IdentityVerdictKind =
  | 'same_person'
  | 'review'
  | 'different_person'
  | 'blocked';

export interface ScoredPair {
  sample: string;
  similarity: number;
  copiedFromBase: boolean;
}

export interface IdentityVerdict {
  method: string;
  compositing: false;
  engine: IdentityEngine;
  rtxAvailable: boolean;
  baseCount: number;
  sampleCount: number;
  scoredCount: number;
  meanSimilarity: number | null;
  minSimilarity: number | null;
  samePerson: boolean;
  verdict: IdentityVerdictKind;
  reason: string;
}

export interface JudgeIdentityInput {
  method: string;
  engine: IdentityEngine;
  rtxAvailable: boolean;
  baseCount: number;
  pairs: ScoredPair[];
}

/**
 * Decide whether LoRA text-to-image samples are the same person as the base set.
 * A copied base file, a composite method, or a non-ArcFace embedding cannot pass.
 */
export function judgeIdentity(input: JudgeIdentityInput): IdentityVerdict {
  const compositing = false as const;
  const usable = input.pairs.filter((pair) => !pair.copiedFromBase);
  const copied = input.pairs.length - usable.length;
  const meanSimilarity = mean(usable.map((pair) => pair.similarity));
  const minSimilarity = usable.length ? Math.min(...usable.map((pair) => pair.similarity)) : null;

  const base: IdentityVerdict = {
    method: input.method,
    compositing,
    engine: input.engine,
    rtxAvailable: input.rtxAvailable,
    baseCount: input.baseCount,
    sampleCount: input.pairs.length,
    scoredCount: usable.length,
    meanSimilarity,
    minSimilarity,
    samePerson: false,
    verdict: 'blocked',
    reason: 'blocked',
  };

  if (input.method !== LORA_GENERATION_METHOD || methodIsForbidden(input.method)) {
    return { ...base, reason: 'face_composite_forbidden' };
  }

  if (input.baseCount === 0) {
    return { ...base, reason: 'missing_base_images' };
  }

  if (input.pairs.length > 0 && usable.length === 0) {
    return { ...base, reason: 'sample_is_copy_of_base' };
  }

  if (copied > 0 && usable.length === 0) {
    return { ...base, reason: 'sample_is_copy_of_base' };
  }

  if (input.engine !== 'arcface') {
    if (usable.length === 0) {
      return {
        ...base,
        reason: input.rtxAvailable ? 'samples_missing' : 'rtx_required',
      };
    }
    return {
      ...base,
      verdict: 'review',
      reason: 'proxy_embedding_is_not_identity_proof',
    };
  }

  if (usable.length === 0) {
    return {
      ...base,
      reason: input.rtxAvailable ? 'samples_missing' : 'rtx_required',
    };
  }

  if (
    meanSimilarity !== null &&
    minSimilarity !== null &&
    meanSimilarity >= LORA_IDENTITY_THRESHOLDS.sameMean &&
    minSimilarity >= LORA_IDENTITY_THRESHOLDS.sameMin
  ) {
    return {
      ...base,
      samePerson: true,
      verdict: 'same_person',
      reason: 'arcface_same_person',
    };
  }

  if (meanSimilarity !== null && meanSimilarity >= LORA_IDENTITY_THRESHOLDS.reviewMean) {
    return { ...base, verdict: 'review', reason: 'arcface_borderline' };
  }

  return { ...base, verdict: 'different_person', reason: 'arcface_different_person' };
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  const sum = values.reduce((acc, value) => acc + value, 0);
  return sum / values.length;
}
