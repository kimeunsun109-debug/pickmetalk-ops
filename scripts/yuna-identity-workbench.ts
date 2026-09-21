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
  const py = join(REPO, 'scripts', 'face-composite', 'pass-loop.py');
  const r = spawnSync(
    'python3',
    [py, '--lock', lock, '--targets', ...targets, '--output-dir', METHOD_A, '--min-pass', '0'],
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

function methodD(lock: string, compositeDir: string) {
  mkdirSync(METHOD_D, { recursive: true });
  const passLoop = join(REPO, 'scripts', 'face-composite', 'pass-loop.py');
  const targets = existsSync(compositeDir)
    ? readdirSync(compositeDir)
        .filter((f) => /_v8\.(jpg|jpeg|png)$/i.test(f))
        .map((f) => join(compositeDir, f))
    : [];

  if (targets.length === 0) {
    const empty = {
      lock,
      method: 'landmark_aligned_face_crop',
      note: 'Run Method A first',
      results: [],
      passThreshold: 0.85,
      passed: 0,
      total: 0,
    };
    writeFileSync(join(METHOD_D, 'qa-report.json'), JSON.stringify(empty, null, 2), 'utf-8');
    console.log(`✓ Method D: ${METHOD_D}/qa-report.json (0/0 pass)`);
    return;
  }

  const r = spawnSync(
    'python3',
    [passLoop, '--lock', lock, '--targets', ...targets, '--output-dir', compositeDir, '--min-pass', '0'],
    { cwd: REPO, encoding: 'utf-8' },
  );
  const reportPath = join(compositeDir, 'pass-loop-report.json');
  if (existsSync(reportPath)) {
    const report = JSON.parse(readFileSync(reportPath, 'utf-8'));
    const qa = {
      lock,
      method: report.qa?.method ?? 'landmark_aligned_face_crop',
      note: 'MediaPipe-aligned face crop QA (upgrade to InsightFace in production)',
      passThreshold: report.pass_threshold ?? 0.85,
      passed: report.qa_passed ?? 0,
      total: report.qa_total ?? 0,
      results: (report.qa?.results ?? []).map((row: { path: string; similarity: number; percent: number; passed: boolean }) => ({
        file: basename(row.path),
        similarity: row.similarity,
        percent: row.percent,
        passed: row.passed,
      })),
    };
    writeFileSync(join(METHOD_D, 'qa-report.json'), JSON.stringify(qa, null, 2), 'utf-8');
    console.log(`✓ Method D: ${METHOD_D}/qa-report.json (${qa.passed}/${qa.total} pass)`);
    return;
  }
  console.error(r.stderr || r.stdout);
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
  methodD(lock, METHOD_A);

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
