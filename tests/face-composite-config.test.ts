import { describe, expect, it } from 'vitest';
import {
  FACE_COMPOSITE_PATHS,
  FACE_LOCK_CONFIGS,
  resolveLockPrimaryPath,
} from '../src/config/face-composite.config.js';

describe('face-composite config', () => {
  it('defines lock configs for all main characters', () => {
    expect(FACE_LOCK_CONFIGS.map((c) => c.character)).toEqual(
      expect.arrayContaining(['yuna', 'narin', 'yunseo', 'eunha', 'jiyu']),
    );
    for (const cfg of FACE_LOCK_CONFIGS) {
      expect(cfg.lockFileName).toBe('01_front_main.jpg');
    }
  });

  it('resolves yuna dev fallback when master library absent', () => {
    const path = resolveLockPrimaryPath('yuna');
    expect(path).toBeTruthy();
    expect(path).toMatch(/selfie|01_front_main/);
  });

  it('exposes composite script paths', () => {
    expect(FACE_COMPOSITE_PATHS.scriptDir).toContain('face-composite');
    expect(FACE_COMPOSITE_PATHS.pythonModule).toContain('composite.py');
  });
});
