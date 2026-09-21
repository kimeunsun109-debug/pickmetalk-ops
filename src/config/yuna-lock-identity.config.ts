/**
 * 유나 LOCK PRIMARY (01_front_main) — 픽셀 기준 identity 정의
 * 제공된 정면 레퍼런스 기준. 재생성 프롬프트·MJ·합성 QA에 공통 사용.
 */
import { join } from 'path';

export const YUNA_LOCK_PATHS = {
  primaryJpg: join(process.cwd(), 'reference', 'yuna', '01_front_main.jpg'),
  primaryPng: join(process.cwd(), 'reference', 'yuna', '01_front_main.png'),
};

/** LOCK 픽셀과 1:1 맞춰야 하는 얼굴 특징 (텍스트는 보조) */
export const YUNA_LOCK_VISUAL = {
  faceShape: 'oval face, soft delicate jawline, small rounded chin',
  eyes: 'large dark brown almond eyes, double eyelid, warm peach eyeshadow, long fine lashes',
  nose: 'slim straight nose bridge, small slightly upturned tip, soft highlight on bridge',
  mouth: 'full lips, glossy peach-coral tint, subtle parted smile showing upper teeth edge',
  skin: 'porcelain dewy skin, soft peach blush on cheek apples, natural pores minimal',
  hair: 'long straight dark brown black hair, fine wispy strands across forehead and cheeks',
  outfit: 'black ribbed knit top at neckline',
} as const;

/** MJ / ComfyUI / Image Factory 공통 positive identity block */
export const YUNA_LOCK_IDENTITY_PROMPT = [
  'SAME PERSON every photo — Yuna, 22 year old Korean woman',
  YUNA_LOCK_VISUAL.faceShape,
  YUNA_LOCK_VISUAL.eyes,
  YUNA_LOCK_VISUAL.nose,
  YUNA_LOCK_VISUAL.mouth,
  YUNA_LOCK_VISUAL.skin,
  YUNA_LOCK_VISUAL.hair,
  'NEVER change face shape, eye shape, nose, or bone structure',
  'identical facial identity to LOCK reference 01_front_main',
  'photorealistic, shot on iPhone, no beauty filter',
].join(', ');

export const YUNA_LOCK_NEGATIVE_PROMPT = [
  'different person',
  'different face',
  'face morph',
  'identity drift',
  'western features',
  'blonde hair',
  'blue eyes',
  'short bob',
  'cartoon',
  'anime',
  'illustration',
  '3d render',
  'plastic skin',
  'over-smoothed skin',
  'deformed face',
  'asymmetrical eyes',
  'heavy makeup change',
  'wrong lip color',
].join(', ');
