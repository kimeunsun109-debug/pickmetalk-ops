import { spawnSync } from 'child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import sharp from 'sharp';
import { afterEach, describe, expect, it } from 'vitest';
import {
  FORBIDDEN_IDENTITY_METHODS,
  isRtxDeviceName,
  libraryJoin,
  methodIsForbidden,
  resolveLoraBaseRoot,
  triggerWord,
} from '../src/config/character-lora.config.js';
import { isCompositeArtifactName, scanBaseImages } from '../src/lib/character-lora/base-library.js';
import { buildCaption, prepareDataset } from '../src/lib/character-lora/dataset.js';
import { judgeIdentity } from '../src/lib/character-lora/identity-check.js';

const temps: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), 'pmt-lora-'));
  temps.push(dir);
  return dir;
}

afterEach(() => {
  for (const dir of temps.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

async function writeFace(path: string, color: { r: number; g: number; b: number }): Promise<void> {
  await sharp({
    create: { width: 64, height: 80, channels: 3, background: color },
  })
    .jpeg()
    .toFile(path);
}

describe('character lora config', () => {
  it('points the base folder at the photo library base directory', () => {
    const runtime = process.env.PICKMETALK_RUNTIME;
    const library = process.env.PHOTO_LIBRARY_ROOT;
    const base = process.env.LORA_BASE_ROOT;
    process.env.PICKMETALK_RUNTIME = 'test';
    delete process.env.LORA_BASE_ROOT;
    delete process.env.PHOTO_LIBRARY_ROOT;
    try {
      expect(resolveLoraBaseRoot().replace(/\\/g, '/')).toMatch(/photo-library\/base$/);
      expect(libraryJoin('D:\\PickMeTalk_PhotoLibrary', 'base', 'yuna')).toBe(
        'D:\\PickMeTalk_PhotoLibrary\\base\\yuna'
      );
    } finally {
      if (runtime === undefined) delete process.env.PICKMETALK_RUNTIME;
      else process.env.PICKMETALK_RUNTIME = runtime;
      if (library === undefined) delete process.env.PHOTO_LIBRARY_ROOT;
      else process.env.PHOTO_LIBRARY_ROOT = library;
      if (base === undefined) delete process.env.LORA_BASE_ROOT;
      else process.env.LORA_BASE_ROOT = base;
    }
  });

  it('recognizes RTX names and rejects face replacement methods', () => {
    expect(isRtxDeviceName('NVIDIA GeForce RTX 4090')).toBe(true);
    expect(isRtxDeviceName('Tesla T4')).toBe(false);
    expect(methodIsForbidden('lora-txt2img')).toBe(false);
    expect(methodIsForbidden('inswapper')).toBe(true);
    expect(methodIsForbidden('face-composite-v8')).toBe(true);
    expect(triggerWord('yuna')).toBe('pmt_yuna');
    expect(FORBIDDEN_IDENTITY_METHODS).toContain('reactor');
  });
});

describe('base dataset', () => {
  it('copies midjourney base faces and skips composite artifacts', async () => {
    const root = tempDir();
    const base = join(root, 'base', 'yuna');
    mkdirSync(base, { recursive: true });
    await writeFace(join(base, 'front.jpg'), { r: 210, g: 170, b: 150 });
    await writeFace(join(base, 'left.jpg'), { r: 200, g: 160, b: 140 });
    await writeFace(join(base, 'scene_v8.jpg'), { r: 10, g: 10, b: 10 });
    writeFileSync(
      join(base, 'swapped.jpg'),
      readFileSync(join(base, 'front.jpg'))
    );
    writeFileSync(
      join(base, 'swapped.meta.json'),
      JSON.stringify({ method: 'face-swap', generatedBy: 'reactor' })
    );
    await writeFace(join(base, 'swapped.jpg'), { r: 1, g: 2, b: 3 });

    const scan = scanBaseImages('yuna', join(root, 'base'));
    expect(scan.images.map((img) => img.filename).sort()).toEqual(['front.jpg', 'left.jpg']);
    expect(scan.skippedComposites.length).toBe(2);
    expect(isCompositeArtifactName('scene_v8.jpg')).toBe(true);

    const prepared = prepareDataset('yuna', {
      baseRoot: join(root, 'base'),
      outputRoot: join(root, 'out'),
    });
    expect(prepared.ok).toBe(true);
    expect(prepared.manifest.compositing).toBe(false);
    expect(prepared.manifest.method).toBe('lora-txt2img');
    expect(prepared.manifest.readyToTrain).toBe(false);
    expect(prepared.manifest.caption.startsWith('pmt_yuna')).toBe(true);
    expect(prepared.manifest.caption.toLowerCase()).not.toContain('inswapper');
    const captionFile = readFileSync(join(prepared.manifest.datasetDir, '001.txt'), 'utf8');
    expect(captionFile).toContain(buildCaption('yuna'));
    const manifest = JSON.parse(readFileSync(join(root, 'out', 'yuna', 'manifest.json'), 'utf8'));
    expect(manifest.train.modelId).toContain('stable-diffusion-xl');
  });
});

describe('identity verdict', () => {
  it('confirms the same person only from arcface scores of generated samples', () => {
    const same = judgeIdentity({
      method: 'lora-txt2img',
      engine: 'arcface',
      rtxAvailable: true,
      baseCount: 8,
      pairs: [
        { sample: 'a.png', similarity: 0.52, copiedFromBase: false },
        { sample: 'b.png', similarity: 0.47, copiedFromBase: false },
      ],
    });
    expect(same.samePerson).toBe(true);
    expect(same.verdict).toBe('same_person');

    const different = judgeIdentity({
      method: 'lora-txt2img',
      engine: 'arcface',
      rtxAvailable: true,
      baseCount: 8,
      pairs: [{ sample: 'a.png', similarity: 0.12, copiedFromBase: false }],
    });
    expect(different.samePerson).toBe(false);
    expect(different.verdict).toBe('different_person');
  });

  it('does not treat a composite, a copy, or a pixel proxy as the same person', () => {
    const composite = judgeIdentity({
      method: 'face-composite-v8',
      engine: 'arcface',
      rtxAvailable: true,
      baseCount: 8,
      pairs: [{ sample: 'a.png', similarity: 0.99, copiedFromBase: false }],
    });
    expect(composite.samePerson).toBe(false);
    expect(composite.reason).toBe('face_composite_forbidden');

    const copied = judgeIdentity({
      method: 'lora-txt2img',
      engine: 'arcface',
      rtxAvailable: true,
      baseCount: 8,
      pairs: [{ sample: 'a.png', similarity: 1, copiedFromBase: true }],
    });
    expect(copied.reason).toBe('sample_is_copy_of_base');
    expect(copied.samePerson).toBe(false);

    const proxy = judgeIdentity({
      method: 'lora-txt2img',
      engine: 'proxy',
      rtxAvailable: false,
      baseCount: 8,
      pairs: [{ sample: 'a.png', similarity: 0.99, copiedFromBase: false }],
    });
    expect(proxy.samePerson).toBe(false);
    expect(proxy.reason).toBe('proxy_embedding_is_not_identity_proof');

    const blocked = judgeIdentity({
      method: 'lora-txt2img',
      engine: 'none',
      rtxAvailable: false,
      baseCount: 8,
      pairs: [],
    });
    expect(blocked.reason).toBe('rtx_required');
    expect(blocked.samePerson).toBe(false);
  });
});

describe('lora scripts refuse image conditioning', () => {
  it('rejects an init image before any model load', () => {
    const result = spawnSync(
      'python3',
      ['scripts/lora/sample_txt2img.py', '--manifest', 'missing.json', '--init-image', 'face.png'],
      { encoding: 'utf8', cwd: process.cwd() }
    );
    expect(result.status).toBe(2);
    expect(`${result.stdout}\n${result.stderr}`).toContain('text-to-image only');
  });

  it('rejects a composite training manifest', () => {
    const dir = tempDir();
    const manifest = join(dir, 'manifest.json');
    writeFileSync(
      manifest,
      JSON.stringify({ method: 'face-swap', compositing: true, readyToTrain: true, imageCount: 8 })
    );
    const result = spawnSync('python3', ['scripts/lora/train_sdxl_lora.py', '--manifest', manifest, '--check'], {
      encoding: 'utf8',
      cwd: process.cwd(),
    });
    expect(result.status).toBe(2);
    expect(`${result.stdout}\n${result.stderr}`).toContain('without compositing');
  });
});
