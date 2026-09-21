import { existsSync } from 'fs';
import { join } from 'path';
import { PHOTO_LIBRARY_ROOT } from './photo-universe.config.js';

/** Immutable LOCK PRIMARY face reference per character. */
export interface FaceLockConfig {
  character: string;
  /** Canonical filename e.g. 01_front_main.jpg */
  lockFileName: string;
  /** Optional override via env FACE_LOCK_YUNA etc. */
  lockPath?: string;
}

export const FACE_LOCK_CONFIGS: FaceLockConfig[] = [
  { character: 'yuna', lockFileName: '01_front_main.jpg' },
  { character: 'narin', lockFileName: '01_front_main.jpg' },
  { character: 'yunseo', lockFileName: '01_front_main.jpg' },
  { character: 'eunha', lockFileName: '01_front_main.jpg' },
  { character: 'jiyu', lockFileName: '01_front_main.jpg' },
];

const REPO_ROOT = process.cwd();

export function resolveLockPrimaryPath(character: string): string | null {
  const cfg = FACE_LOCK_CONFIGS.find((c) => c.character === character);
  if (!cfg) return null;

  const envKey = `FACE_LOCK_${character.toUpperCase()}`;
  const envPath = process.env[envKey]?.trim();
  if (envPath && existsSync(envPath)) return envPath;

  const candidates = [
    join(PHOTO_LIBRARY_ROOT, 'master', character, cfg.lockFileName),
    join(PHOTO_LIBRARY_ROOT, 'master', character, '01_front_main.png'),
    join(REPO_ROOT, 'reference', character, cfg.lockFileName),
    join(REPO_ROOT, 'reference', character, '01_front_main.png'),
    // Dev fallback when master library not mounted
    join(REPO_ROOT, 'assets', 'photos', character, 'selfie', 'f39b903652c76021.jpg'),
  ];

  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return null;
}

export const FACE_COMPOSITE_PATHS = {
  scriptDir: join(REPO_ROOT, 'scripts', 'face-composite'),
  pythonModule: join(REPO_ROOT, 'scripts', 'face-composite', 'v8', 'composite.py'),
  defaultOutputDir: join(REPO_ROOT, 'data', 'face-composite', 'output'),
  debugDir: join(REPO_ROOT, 'data', 'face-composite', 'debug'),
};
