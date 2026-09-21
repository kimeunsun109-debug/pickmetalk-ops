#!/usr/bin/env tsx
/**
 * CLI: composite LOCK PRIMARY (01_front_main) onto a target body image.
 *
 * Usage:
 *   npm run face:composite -- --character=yuna --target=path/to/body.jpg
 *   npm run face:composite -- --lock=ref/01_front_main.jpg --target=body.jpg --output=out.jpg
 */
import { spawnSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { basename, isAbsolute, join, resolve } from 'path';
import {
  FACE_COMPOSITE_PATHS,
  resolveLockPrimaryPath,
} from '../src/config/face-composite.config.js';

function parseArgs(argv: string[]) {
  const out: Record<string, string> = {};
  for (const arg of argv) {
    const m = arg.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2];
  }
  return out;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const character = args.character ?? 'yuna';
  const lock = args.lock ?? resolveLockPrimaryPath(character);
  const target = args.target;

  if (!lock || !existsSync(lock)) {
    console.error(
      `LOCK PRIMARY not found for "${character}". Set --lock= or FACE_LOCK_${character.toUpperCase()}=`,
    );
    process.exit(1);
  }
  if (!target || !existsSync(target)) {
    console.error('Missing --target=<body-image-path>');
    process.exit(1);
  }

  const repoRoot = process.cwd();
  const absLock = isAbsolute(lock) ? lock : resolve(repoRoot, lock);
  const absTarget = isAbsolute(target) ? target : resolve(repoRoot, target);
  const output =
    args.output ??
    join(
      FACE_COMPOSITE_PATHS.defaultOutputDir,
      character,
      `${basename(target, '.jpg')}_v8.jpg`,
    );
  const absOutput = isAbsolute(output) ? output : resolve(repoRoot, output);
  const debugDir = args.debug ? FACE_COMPOSITE_PATHS.debugDir : undefined;
  const absDebugDir = debugDir
    ? join(debugDir, basename(target, '.jpg'))
    : undefined;

  mkdirSync(join(absOutput, '..'), { recursive: true });
  if (absDebugDir) mkdirSync(absDebugDir, { recursive: true });

  const pyArgs = [
    FACE_COMPOSITE_PATHS.pythonModule,
    '--lock',
    absLock,
    '--target',
    absTarget,
    '--output',
    absOutput,
    '--color-match',
    args['color-match'] ?? '0.65',
    '--hair-strength',
    args['hair-strength'] ?? '0.92',
    '--frontal',
    args.frontal ?? 'auto',
  ];
  if (absDebugDir) {
    pyArgs.push('--debug-dir', absDebugDir);
  }

  const v8Dir = join(FACE_COMPOSITE_PATHS.scriptDir, 'v8');
  const result = spawnSync('python3', pyArgs.map((a) => (a === FACE_COMPOSITE_PATHS.pythonModule ? 'composite.py' : a)), {
    cwd: v8Dir,
    stdio: 'inherit',
  });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
  console.log(`\n✓ v8 composite: ${absOutput}`);
}

main();
