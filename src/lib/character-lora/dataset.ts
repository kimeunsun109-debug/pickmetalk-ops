import { copyFileSync, mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { getCharacterFaceIdentity } from '../../config/character-face-reference.config.js';
import {
  datasetRepeats,
  defaultTrainConfig,
  LORA_GENERATION_METHOD,
  LORA_MIN_TRAIN_IMAGES,
  LORA_OUTPUT_ROOT,
  triggerWord,
  type LoraTrainConfig,
} from '../../config/character-lora.config.js';
import type { CharacterSlug } from '../../config/character-face-reference.config.js';
import { scanBaseImages, type BaseImage } from './base-library.js';

export interface LoraManifest {
  character: CharacterSlug;
  method: typeof LORA_GENERATION_METHOD;
  compositing: false;
  trigger: string;
  baseRoot: string;
  datasetDir: string;
  outputDir: string;
  imageCount: number;
  repeats: number;
  readyToTrain: boolean;
  minTrainImages: number;
  skippedComposites: string[];
  negativePrompt: string;
  prompts: string[];
  train: LoraTrainConfig;
  baseImages: Array<{ path: string; filename: string; sha256: string }>;
  caption: string;
}

export interface PrepareDatasetResult {
  ok: boolean;
  reason: string;
  manifest: LoraManifest;
}

const NEGATIVE_PROMPT = [
  'different person',
  'different face',
  'wrong identity',
  'face swap',
  'composite face',
  'plastic skin',
  'cartoon',
  'anime',
  'deformed face',
].join(', ');

export function buildCaption(slug: CharacterSlug): string {
  const identity = getCharacterFaceIdentity(slug);
  const trigger = triggerWord(slug);
  const lock = identity?.identityPrompt ?? slug;
  return `${trigger}, ${lock}, photorealistic natural smartphone photo, realistic skin texture`;
}

export function verificationPrompts(slug: CharacterSlug): string[] {
  const caption = buildCaption(slug);
  return [
    `${caption}, close-up smartphone selfie, natural window light, looking at camera`,
    `${caption}, three-quarter view, cafe, casual clothes, natural daylight`,
    `${caption}, outdoor daylight, slight smile, shoulders-up portrait`,
  ];
}

function datasetFileName(index: number, filename: string): string {
  const ext = filename.match(/\.(jpg|jpeg|png|webp)$/i)?.[0] ?? '.jpg';
  return `${String(index + 1).padStart(3, '0')}${ext.toLowerCase()}`;
}

export function prepareDataset(
  slug: CharacterSlug,
  options?: { baseRoot?: string; outputRoot?: string }
): PrepareDatasetResult {
  const scan = scanBaseImages(slug, options?.baseRoot);
  const outputDir = join(options?.outputRoot ?? LORA_OUTPUT_ROOT, slug);
  const repeats = datasetRepeats(scan.images.length);
  const trigger = triggerWord(slug);
  const datasetDir = join(outputDir, 'dataset', `${repeats}_${trigger}`);
  const caption = buildCaption(slug);

  const manifest: LoraManifest = {
    character: slug,
    method: LORA_GENERATION_METHOD,
    compositing: false,
    trigger,
    baseRoot: scan.baseRoot,
    datasetDir,
    outputDir,
    imageCount: scan.images.length,
    repeats,
    readyToTrain: scan.images.length >= LORA_MIN_TRAIN_IMAGES,
    minTrainImages: LORA_MIN_TRAIN_IMAGES,
    skippedComposites: scan.skippedComposites,
    negativePrompt: NEGATIVE_PROMPT,
    prompts: verificationPrompts(slug),
    train: defaultTrainConfig(),
    baseImages: scan.images.map((img: BaseImage) => ({
      path: img.path,
      filename: img.filename,
      sha256: img.sha256,
    })),
    caption,
  };

  if (scan.images.length === 0) {
    return {
      ok: false,
      reason: 'missing_base_images',
      manifest,
    };
  }

  mkdirSync(datasetDir, { recursive: true });
  scan.images.forEach((img, index) => {
    const name = datasetFileName(index, img.filename);
    const stem = name.replace(/\.(jpg|jpeg|png|webp)$/i, '');
    copyFileSync(img.path, join(datasetDir, name));
    writeFileSync(join(datasetDir, `${stem}.txt`), `${caption}\n`, 'utf8');
  });

  writeFileSync(join(outputDir, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf8');
  return { ok: true, reason: manifest.readyToTrain ? 'ready' : 'too_few_images', manifest };
}
