import { execFileSync } from 'child_process';
import { isRtxDeviceName } from '../../config/character-lora.config.js';

export interface RtxStatus {
  available: boolean;
  device: string | null;
  reason: 'ok' | 'nvidia_smi_missing' | 'cuda_gpu_is_not_rtx' | 'no_gpu_listed';
}

export function detectRtx(): RtxStatus {
  let output = '';
  try {
    output = execFileSync('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], {
      encoding: 'utf8',
      timeout: 8000,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
  } catch {
    return { available: false, device: null, reason: 'nvidia_smi_missing' };
  }

  const device = output.split(/\r?\n/).map((line) => line.trim()).find((line) => line.length > 0) ?? '';
  if (!device) return { available: false, device: null, reason: 'no_gpu_listed' };
  if (!isRtxDeviceName(device)) {
    return { available: false, device, reason: 'cuda_gpu_is_not_rtx' };
  }
  return { available: true, device, reason: 'ok' };
}
