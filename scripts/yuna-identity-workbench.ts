#!/usr/bin/env tsx
/**
 * 유나 얼굴 일관성 — 4가지 방법 한번에 실행
 *
 * Method A: v8 pixel LOCK 합성 (identity 보장)
 * Method B: MJ --cref 명령 생성 (생성 단계 face lock)
 * Method C: ComfyUI/SDXL 프롬프트 팩 (텍스트 identity lock)
 * Method D: Face QA embedding (LOCK 기준 유사도)
 *
 * Usage:
 *   npm run yuna:identity
 *   npm run yuna:identity -- --skip-composite
 */
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'fs';
import { basename, join, resolve } from 'path';
import { buildCharacterMjCommand } from '../src/config/character-face-reference.config.js';
import { resolveLockPrimaryPath } from '../src/config/face-composite.config.js';
import {
  YUNA_LOCK_IDENTITY_PROMPT,
  YUNA_LOCK_NEGATIVE_PROMPT,
  YUNA_LOCK_PATHS,
} from '../src/config/yuna-lock-identity.config.js';
import { extractFaceEmbedding, cosineSimilarity } from '../src/lib/midjourney-production/face-verifier.js';

const REPO = process.cwd();
const OUT = join(REPO, 'data', 'yuna-identity');
const METHOD_A = join(OUT, 'method-a-v8-composite');
const METHOD_B = join(OUT, 'method-b-mj-cref');
const METHOD_C = join(OUT, 'method-c-prompt-pack');
const METHOD_D = join(OUT, 'method-d-face-qa');

function parseArgs(argv: string[]) {
  return {
    skipComposite: argv.includes('--skip-composite'),
    scenes: argv.filter((a) => !a.startsWith('--')),
  };
}

function findYunaBodyTargets(): string[] {
  const root = join(REPO, 'assets', 'photos', 'yuna');
  const lock = resolveLockPrimaryPath('yuna') ?? YUNA_LOCK_PATHS.primaryJpg;
  const lockBase = basename(lock);
  const files: string[] = [];

  function walk(dir: string) {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (/\.(jpg|jpeg|png|webp)$/i.test(e.name) && full !== lock) files.push(full);
    }
  }
  if (existsSync(root)) walk(root);
  return files.sort();
}

function methodA(lock: string, targets: string[]) {
  mkdirSync(METHOD_A, { recursive: true });
  const py = join(REPO, 'scripts', 'face-composite', 'batch-composite.py');
  const r = spawnSync(
    'python3',
    [py, '--lock', lock, '--targets', ...targets, '--output-dir', METHOD_A],
    { cwd: REPO, stdio: 'inherit' },
  );
  return r.status === 0;
}

function methodB() {
  mkdirSync(METHOD_B, { recursive: true });
  const scenes = [
    'gentle smile, soft indoor window light, beige knit sweater, campus cafe background',
    'natural laugh, outdoor daylight, hair down, Korean street style',
    'cozy weekend at home, grey hoodie, warm lamp light',
    'rainy day with umbrella, soft overcast light, holding coffee cup',
    'morning selfie, bed hair, natural bedroom light',
  ];
  const lines: string[] = [
    '# Yuna — MJ Identity Lock Commands (--cref)',
    '',
    `LOCK file: \`${YUNA_LOCK_PATHS.primaryJpg}\``,
    'Upload LOCK to Discord/CDN, then set `MJ_CREF_YUNA=<url>` in `.env`.',
    '',
    '```env',
    'MJ_CREF_YUNA=https://your-cdn/yuna_01_front_main.jpg',
    'MJ_CREF_WEIGHT=100',
    '```',
    '',
  ];
  scenes.forEach((scene, i) => {
    const cmd = buildCharacterMjCommand('yuna', scene, 'blurry, low quality, watermark');
    lines.push(`## Scene ${i + 1}`, '', '```', cmd, '```', '');
  });
  writeFileSync(join(METHOD_B, 'YUNA_MJ_CREF_COMMANDS.md'), lines.join('\n'), 'utf-8');
  console.log(`✓ Method B: ${METHOD_B}/YUNA_MJ_CREF_COMMANDS.md`);
}

function methodC() {
  mkdirSync(METHOD_C, { recursive: true });
  const pack = {
    character: 'yuna',
    lockPrimary: YUNA_LOCK_PATHS.primaryJpg,
    methods: {
      pixelLock: 'npm run face:composite -- --character=yuna --target=<body.jpg>',
      mjCref: 'MJ_CREF_YUNA + buildCharacterMjCommand',
      comfyui: 'IP-Adapter Face / reference image = 01_front_main (no InstantID redraw)',
    },
    positive: YUNA_LOCK_IDENTITY_PROMPT,
    negative: YUNA_LOCK_NEGATIVE_PROMPT,
    comfyuiWorkflowHints: [
      'Load Image → 01_front_main.jpg as IPAdapter face reference',
      'Keep denoise ≤ 0.45 for img2img body-only passes',
      'Do NOT use face swap nodes that regenerate identity',
      'Post: npm run face:composite for pixel-perfect LOCK overlay',
    ],
    scenes: [
      { id: 'cafe', prompt: `${YUNA_LOCK_IDENTITY_PROMPT}, sitting in cafe, holding latte` },
      { id: 'campus', prompt: `${YUNA_LOCK_IDENTITY_PROMPT}, university campus walkway, backpack` },
      { id: 'home', prompt: `${YUNA_LOCK_IDENTITY_PROMPT}, cozy living room, weekend relax` },
    ],
  };
  writeFileSync(join(METHOD_C, 'yuna-identity-prompt-pack.json'), JSON.stringify(pack, null, 2), 'utf-8');
  console.log(`✓ Method C: ${METHOD_C}/yuna-identity-prompt-pack.json`);
}

async function methodD(lock: string, compositeDir: string) {
  mkdirSync(METHOD_D, { recursive: true });
  const lockEmb = await extractFaceEmbedding(lock);
  const rows: Array<{ file: string; similarity: number; percent: number }> = [];

  if (existsSync(compositeDir)) {
    for (const f of readdirSync(compositeDir)) {
      if (!/\.(jpg|jpeg|png)$/i.test(f)) continue;
      const p = join(compositeDir, f);
      const emb = await extractFaceEmbedding(p);
      const sim = cosineSimilarity(lockEmb, emb);
      rows.push({ file: f, similarity: sim, percent: Math.round(sim * 1000) / 10 });
    }
  }

  rows.sort((a, b) => b.similarity - a.similarity);
  const report = {
    lock,
    method: 'heuristic_face_region_embedding',
    note: 'Production should upgrade to InsightFace; this is ops smoke QA',
    results: rows,
    passThreshold: 0.85,
    passed: rows.filter((r) => r.similarity >= 0.85).length,
    total: rows.length,
  };
  writeFileSync(join(METHOD_D, 'qa-report.json'), JSON.stringify(report, null, 2), 'utf-8');
  console.log(`✓ Method D: ${METHOD_D}/qa-report.json (${report.passed}/${report.total} pass)`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const lock = resolveLockPrimaryPath('yuna') ?? YUNA_LOCK_PATHS.primaryJpg;

  if (!existsSync(lock)) {
    console.error(`LOCK not found: ${lock}`);
    process.exit(1);
  }

  mkdirSync(OUT, { recursive: true });
  console.log(`\n유나 LOCK: ${lock}\n`);

  const targets = findYunaBodyTargets();
  console.log(`Body targets: ${targets.length} images\n`);

  if (!args.skipComposite && targets.length > 0) {
    console.log('--- Method A: v8 pixel LOCK composite ---');
    methodA(lock, targets);
  } else if (args.skipComposite) {
    console.log('--- Method A: skipped (--skip-composite) ---');
  }

  console.log('\n--- Method B: MJ --cref commands ---');
  methodB();

  console.log('\n--- Method C: prompt pack ---');
  methodC();

  console.log('\n--- Method D: face QA ---');
  await methodD(lock, METHOD_A);

  const readme = [
    '# Yuna Identity Workbench Output',
    '',
    '| Method | Folder | Role |',
    '|--------|--------|------|',
    '| A | method-a-v8-composite | **Identity PASS** — LOCK 픽셀 합성 |',
    '| B | method-b-mj-cref | MJ 생성 시 --cref face lock |',
    '| C | method-c-prompt-pack | ComfyUI/SDXL 텍스트 identity |',
    '| D | method-d-face-qa | LOCK 대비 유사도 QA |',
    '',
    '**권장 운영**: body는 B/C로 생성 → 최종 identity는 A로 LOCK 합성.',
  ].join('\n');
  writeFileSync(join(OUT, 'README.md'), readme, 'utf-8');
  console.log(`\n✓ Done → ${OUT}/`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
