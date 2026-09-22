/**
 * Character LoRA — Midjourney base faces, RTX text-to-image only.
 * Face paste / swap / blend is out of scope. Identity is learned, then checked.
 */
import { join } from 'path';
import { CHARACTER_SLUGS, type CharacterSlug } from './character-face-reference.config.js';
import { resolvePhotoLibraryRoot } from './runtime-environment.config.js';

export const LORA_BASE_FOLDER = 'base';

/** Generation path this pipeline is allowed to use. */
export const LORA_GENERATION_METHOD = 'lora-txt2img' as const;

/**
 * Anything that pastes, swaps, or conditions on a face image is forbidden.
 * Matched case-insensitively against file names, sidecars, and method strings.
 */
export const FORBIDDEN_IDENTITY_METHODS = [
  'composite',
  'face-swap',
  'face_swap',
  'faceswap',
  'inswapper',
  'reactor',
  'facefusion',
  'ip-adapter',
  'ipadapter',
  'instantid',
  'photomaker',
  'faceid',
  'paste',
  'blend',
] as const;

/** ArcFace cosine on normed embeddings. Proxy pixels never confirm identity. */
export const LORA_IDENTITY_THRESHOLDS = {
  sameMean: 0.4,
  sameMin: 0.32,
  reviewMean: 0.3,
} as const;

export const LORA_MIN_TRAIN_IMAGES = 5;

export interface LoraTrainConfig {
  modelId: string;
  rank: number;
  alpha: number;
  learningRate: number;
  maxSteps: number;
  resolution: number;
}

export function defaultTrainConfig(): LoraTrainConfig {
  return {
    modelId: process.env.LORA_MODEL_ID ?? 'stabilityai/stable-diffusion-xl-base-1.0',
    rank: Number(process.env.LORA_RANK ?? 32),
    alpha: Number(process.env.LORA_ALPHA ?? 16),
    learningRate: Number(process.env.LORA_LR ?? 1e-4),
    maxSteps: Number(process.env.LORA_MAX_STEPS ?? 800),
    resolution: Number(process.env.LORA_RESOLUTION ?? 1024),
  };
}

export const LORA_OUTPUT_ROOT =
  process.env.LORA_OUTPUT_ROOT ?? join(process.cwd(), 'data', 'lora');

export function triggerWord(slug: string): string {
  return `pmt_${slug}`;
}

export function isCharacterSlug(slug: string): slug is CharacterSlug {
  return (CHARACTER_SLUGS as readonly string[]).includes(slug);
}

/** Join a library root without letting Linux `path.join` swallow Windows drive letters. */
export function libraryJoin(root: string, ...parts: string[]): string {
  const sep = root.includes('\\') ? '\\' : '/';
  const head = root.replace(/[\\/]+$/, '');
  return [head, ...parts.filter((p) => p.length > 0)].join(sep);
}

/** `D:\PickMeTalk_PhotoLibrary\base` in production, test-fixtures equivalent otherwise. */
export function resolveLoraBaseRoot(): string {
  const override = process.env.LORA_BASE_ROOT?.trim();
  if (override) return override;
  return libraryJoin(resolvePhotoLibraryRoot(), LORA_BASE_FOLDER);
}

export function datasetRepeats(imageCount: number): number {
  if (imageCount >= 30) return 5;
  if (imageCount >= 15) return 8;
  if (imageCount >= 8) return 12;
  return 20;
}

export function isRtxDeviceName(name: string): boolean {
  return /\bRTX\b/i.test(name);
}

export function methodIsForbidden(method: string): boolean {
  if (method.trim().toLowerCase() === LORA_GENERATION_METHOD) return false;
  const hay = method.toLowerCase();
  return FORBIDDEN_IDENTITY_METHODS.some((token) => hay.includes(token));
}

/** Folder name aliases so `base/유나` and `base/yuna` both map to the slug. */
export const BASE_FOLDER_ALIASES: Record<string, CharacterSlug> = {
  yuna: 'yuna',
  유나: 'yuna',
  narin: 'narin',
  나린: 'narin',
  yunseo: 'yunseo',
  윤서: 'yunseo',
  eunha: 'eunha',
  은하: 'eunha',
  jiyu: 'jiyu',
  지유: 'jiyu',
};
