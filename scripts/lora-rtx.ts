#!/usr/bin/env npx tsx
/**
 * Prepare a character LoRA from Midjourney base faces and, on an NVIDIA RTX,
 * train + sample with text-to-image only, then score identity with ArcFace.
 *
 * Face compositing is refused. This command never pastes a base face onto a scene.
 */
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { CHARACTER_SLUGS, type CharacterSlug } from '../src/config/character-face-reference.config.js';
import { isCharacterSlug, LORA_OUTPUT_ROOT } from '../src/config/character-lora.config.js';
import { expectedBaseDir, fileSha256, prepareDataset } from '../src/lib/character-lora/index.js';
import { judgeIdentity, type IdentityEngine, type IdentityVerdict, type ScoredPair } from '../src/lib/character-lora/identity-check.js';
import { detectRtx, type RtxStatus } from '../src/lib/character-lora/rtx.js';
import type { LoraManifest } from '../src/lib/character-lora/dataset.js';

type Command = 'prepare' | 'verify' | 'run';

interface ArcfaceReport {
  engine?: string;
  pairs?: Array<{ sample: string; similarity: number }>;
  error?: string;
}

function pythonBin(): string {
  return process.platform === 'win32' ? 'python' : 'python3';
}

function parseArgs(argv: string[]): { command: Command; slugs: CharacterSlug[] } {
  const command: Command =
    argv[0] === 'prepare' || argv[0] === 'verify' || argv[0] === 'run' ? argv[0] : 'run';
  const rest = command === argv[0] ? argv.slice(1) : argv;
  const characterFlag = rest.find((arg) => arg.startsWith('--character='));
  const characterIndex = rest.indexOf('--character');
  const raw =
    characterFlag?.slice('--character='.length) ??
    (characterIndex >= 0 ? rest[characterIndex + 1] : undefined);
  if (rest.includes('--all') || !raw) return { command, slugs: [...CHARACTER_SLUGS] };
  if (!isCharacterSlug(raw)) {
    throw new Error(`unknown character: ${raw}`);
  }
  return { command, slugs: [raw] };
}

function sampleFiles(outputDir: string): string[] {
  const dir = join(outputDir, 'samples');
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => /\.(jpg|jpeg|png|webp)$/i.test(name))
    .map((name) => join(dir, name));
}

function runPython(script: string, args: string[]): { status: number; stdout: string; stderr: string } {
  const result = spawnSync(pythonBin(), [join('scripts', 'lora', script), ...args], {
    encoding: 'utf8',
    cwd: process.cwd(),
  });
  return {
    status: result.status ?? 1,
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? result.error?.message ?? '',
  };
}

function readArcface(reportPath: string): { engine: IdentityEngine; pairs: ScoredPair[] } {
  if (!existsSync(reportPath)) return { engine: 'none', pairs: [] };
  const data = JSON.parse(readFileSync(reportPath, 'utf8')) as ArcfaceReport;
  if (data.engine !== 'arcface' || !data.pairs) return { engine: 'none', pairs: [] };
  return {
    engine: 'arcface',
    pairs: data.pairs.map((pair) => ({
      sample: pair.sample,
      similarity: pair.similarity,
      copiedFromBase: false,
    })),
  };
}

function markCopies(pairs: ScoredPair[], samplePaths: string[], baseHashes: Set<string>): ScoredPair[] {
  const byName = new Map(samplePaths.map((file) => [file.split(/[\\/]/).pop() ?? file, file]));
  return pairs.map((pair) => {
    const file = byName.get(pair.sample);
    const copied = file ? baseHashes.has(fileSha256(file)) : false;
    return { ...pair, copiedFromBase: copied };
  });
}

function blocked(rtx: RtxStatus, baseCount: number, reason: string): IdentityVerdict {
  return {
    method: 'lora-txt2img',
    compositing: false,
    engine: 'none',
    rtxAvailable: rtx.available,
    baseCount,
    sampleCount: 0,
    scoredCount: 0,
    meanSimilarity: null,
    minSimilarity: null,
    samePerson: false,
    verdict: 'blocked',
    reason,
  };
}

function verifyManifest(manifest: LoraManifest, rtx: RtxStatus, trainIfMissing: boolean): IdentityVerdict {
  const baseHashes = new Set(manifest.baseImages.map((img) => img.sha256));
  const outputDir = manifest.outputDir;
  mkdirSync(outputDir, { recursive: true });

  if (trainIfMissing) {
    if (!manifest.readyToTrain) {
      return blocked(rtx, manifest.imageCount, 'too_few_images');
    }
    if (!rtx.available) {
      return blocked(rtx, manifest.imageCount, 'rtx_required');
    }
    const manifestPath = join(outputDir, 'manifest.json');
    const trained = runPython('train_sdxl_lora.py', ['--manifest', manifestPath]);
    if (trained.status !== 0) {
      writeFileSync(join(outputDir, 'train.log'), `${trained.stdout}\n${trained.stderr}`, 'utf8');
      return blocked(rtx, manifest.imageCount, 'train_failed');
    }
    const sampled = runPython('sample_txt2img.py', ['--manifest', manifestPath]);
    if (sampled.status !== 0) {
      writeFileSync(join(outputDir, 'sample.log'), `${sampled.stdout}\n${sampled.stderr}`, 'utf8');
      return blocked(rtx, manifest.imageCount, 'sample_failed');
    }
  }

  const samples = sampleFiles(outputDir);
  const arcfacePath = join(outputDir, 'arcface.json');
  if (samples.length > 0) {
    runPython('verify_identity.py', ['--manifest', join(outputDir, 'manifest.json'), '--out', arcfacePath]);
  }
  const arcface = readArcface(arcfacePath);
  const pairs =
    arcface.engine === 'arcface'
      ? markCopies(arcface.pairs, samples, baseHashes)
      : samples.map((file) => ({
          sample: file,
          similarity: 0,
          copiedFromBase: baseHashes.has(fileSha256(file)),
        }));

  const verdict = judgeIdentity({
    method: manifest.method,
    engine: arcface.engine,
    rtxAvailable: rtx.available,
    baseCount: manifest.imageCount,
    pairs: arcface.engine === 'arcface' ? pairs : pairs.map((pair) => ({ ...pair, similarity: 0 })),
  });

  if (arcface.engine !== 'arcface' && samples.length > 0 && pairs.some((pair) => !pair.copiedFromBase)) {
    return { ...verdict, reason: 'arcface_unavailable', verdict: 'blocked', samePerson: false };
  }
  return verdict;
}

function printVerdict(slug: string, verdict: IdentityVerdict, baseDir: string): void {
  const label =
    verdict.verdict === 'same_person'
      ? '동일 인물'
      : verdict.verdict === 'different_person'
        ? '다른 인물'
        : verdict.verdict === 'review'
          ? '보류'
          : '확인 불가';
  console.log(`\n[${slug}] ${label} (${verdict.reason})`);
  console.log(`  base     : ${baseDir}`);
  console.log(`  method   : ${verdict.method} / compositing=${verdict.compositing}`);
  console.log(`  engine   : ${verdict.engine}`);
  console.log(`  rtx      : ${verdict.rtxAvailable ? 'yes' : 'no'}`);
  console.log(`  images   : base ${verdict.baseCount}, samples ${verdict.sampleCount}`);
  if (verdict.meanSimilarity !== null) {
    console.log(
      `  arcface  : mean ${verdict.meanSimilarity.toFixed(3)}, min ${verdict.minSimilarity?.toFixed(3)}`
    );
  }
}

function main(): void {
  const { command, slugs } = parseArgs(process.argv.slice(2));
  const rtx = detectRtx();
  console.log('PickMeTalk character LoRA — text-to-image only, no face composite');
  console.log(`RTX: ${rtx.available ? rtx.device : rtx.reason}`);

  const reports: Array<{ character: string; verdict: IdentityVerdict }> = [];

  for (const slug of slugs) {
    const prepared = prepareDataset(slug);
    const baseDir = expectedBaseDir(slug);
    if (!prepared.ok) {
      const missing = judgeIdentity({
        method: 'lora-txt2img',
        engine: 'none',
        rtxAvailable: rtx.available,
        baseCount: 0,
        pairs: [],
      });
      reports.push({ character: slug, verdict: missing });
      printVerdict(slug, missing, baseDir);
      continue;
    }

    if (command === 'prepare') {
      const verdict: IdentityVerdict = {
        method: 'lora-txt2img',
        compositing: false,
        engine: 'none',
        rtxAvailable: rtx.available,
        baseCount: prepared.manifest.imageCount,
        sampleCount: 0,
        scoredCount: 0,
        meanSimilarity: null,
        minSimilarity: null,
        samePerson: false,
        verdict: 'blocked',
        reason: prepared.manifest.readyToTrain ? 'dataset_ready' : 'too_few_images',
      };
      reports.push({ character: slug, verdict });
      console.log(`\n[${slug}] dataset ${prepared.manifest.datasetDir} (${prepared.manifest.imageCount} images)`);
      continue;
    }

    const verdict = verifyManifest(prepared.manifest, rtx, command === 'run');
    writeFileSync(
      join(prepared.manifest.outputDir, 'identity-report.json'),
      JSON.stringify({ character: slug, rtx, verdict }, null, 2),
      'utf8'
    );
    reports.push({ character: slug, verdict });
    printVerdict(slug, verdict, baseDir);
  }

  const summaryPath = join(LORA_OUTPUT_ROOT, 'identity-summary.json');
  mkdirSync(LORA_OUTPUT_ROOT, { recursive: true });
  writeFileSync(summaryPath, JSON.stringify({ command, rtx, reports }, null, 2), 'utf8');
  console.log(`\nsummary: ${summaryPath}`);

  const confirmed = reports.filter((item) => item.verdict.samePerson);
  console.log(`same person confirmed: ${confirmed.length}/${reports.length}`);
  if (confirmed.length === reports.length && reports.length > 0) process.exit(0);
  if (reports.some((item) => item.verdict.reason === 'missing_base_images')) process.exit(2);
  if (reports.some((item) => item.verdict.verdict === 'different_person')) process.exit(1);
  process.exit(3);
}

main();
