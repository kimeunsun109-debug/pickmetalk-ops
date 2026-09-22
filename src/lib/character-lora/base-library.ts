import { createHash } from 'crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'fs';
import { basename, join } from 'path';
import {
  BASE_FOLDER_ALIASES,
  FORBIDDEN_IDENTITY_METHODS,
  libraryJoin,
  resolveLoraBaseRoot,
} from '../../config/character-lora.config.js';
import {
  CHARACTER_SLUGS,
  type CharacterSlug,
} from '../../config/character-face-reference.config.js';

const IMAGE_EXT = /\.(jpg|jpeg|png|webp)$/i;

export interface BaseImage {
  slug: CharacterSlug;
  path: string;
  filename: string;
  sha256: string;
}

export interface BaseScan {
  baseRoot: string;
  images: BaseImage[];
  skippedComposites: string[];
}

export function fileSha256(filePath: string): string {
  const data = readFileSync(filePath);
  return createHash('sha256').update(data).digest('hex');
}

export function isCompositeArtifactName(name: string): boolean {
  const hay = name.toLowerCase();
  if (/_v\d+\.(jpg|jpeg|png|webp)$/i.test(name)) return true;
  return FORBIDDEN_IDENTITY_METHODS.some((token) => hay.includes(token));
}

function sidecarForbids(imagePath: string): boolean {
  const sidecar = imagePath.replace(/\.(jpg|jpeg|png|webp)$/i, '.meta.json');
  if (!existsSync(sidecar)) return false;
  try {
    const data = JSON.parse(readFileSync(sidecar, 'utf8')) as Record<string, unknown>;
    const hay = ['method', 'generatedBy', 'pipeline', 'source', 'tool']
      .map((key) => String(data[key] ?? ''))
      .join(' ')
      .toLowerCase();
    return FORBIDDEN_IDENTITY_METHODS.some((token) => hay.includes(token));
  } catch {
    return false;
  }
}

function slugFromFilename(filename: string): CharacterSlug | null {
  const stem = filename.replace(/\.(jpg|jpeg|png|webp)$/i, '');
  const lower = stem.toLowerCase();
  for (const slug of CHARACTER_SLUGS) {
    if (lower === slug || lower.startsWith(`${slug}_`) || lower.startsWith(`${slug}-`)) {
      return slug;
    }
  }
  for (const [alias, slug] of Object.entries(BASE_FOLDER_ALIASES)) {
    if (stem === alias || stem.startsWith(`${alias}_`) || stem.startsWith(`${alias}-`)) {
      return slug;
    }
  }
  return null;
}

function collectFromDir(
  dir: string,
  slug: CharacterSlug,
  images: BaseImage[],
  skipped: string[]
): void {
  if (!existsSync(dir)) return;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      collectFromDir(full, slug, images, skipped);
      continue;
    }
    if (!IMAGE_EXT.test(entry.name)) continue;
    if (isCompositeArtifactName(entry.name) || sidecarForbids(full)) {
      skipped.push(full);
      continue;
    }
    images.push({
      slug,
      path: full,
      filename: entry.name,
      sha256: fileSha256(full),
    });
  }
}

/**
 * Read Midjourney base faces from `{library}/base/{slug}` or `{library}/base/{slug}_*.jpg`.
 * Composite artifacts are skipped and never enter the LoRA dataset.
 */
export function scanBaseImages(slug: CharacterSlug, baseRoot = resolveLoraBaseRoot()): BaseScan {
  const images: BaseImage[] = [];
  const skippedComposites: string[] = [];

  if (existsSync(baseRoot)) {
    const characterDir = libraryJoin(baseRoot, slug);
    collectFromDir(characterDir, slug, images, skippedComposites);

    for (const [alias, mapped] of Object.entries(BASE_FOLDER_ALIASES)) {
      if (mapped !== slug || alias === slug) continue;
      collectFromDir(libraryJoin(baseRoot, alias), slug, images, skippedComposites);
    }

    if (statSafeDir(baseRoot)) {
      for (const entry of readdirSync(baseRoot, { withFileTypes: true })) {
        if (entry.isDirectory()) continue;
        if (!IMAGE_EXT.test(entry.name)) continue;
        if (slugFromFilename(entry.name) !== slug) continue;
        const full = join(baseRoot, entry.name);
        if (isCompositeArtifactName(entry.name) || sidecarForbids(full)) {
          skippedComposites.push(full);
          continue;
        }
        images.push({
          slug,
          path: full,
          filename: entry.name,
          sha256: fileSha256(full),
        });
      }
    }
  }

  const seen = new Set<string>();
  const unique = images.filter((img) => {
    if (seen.has(img.sha256)) return false;
    seen.add(img.sha256);
    return true;
  });

  return { baseRoot, images: unique, skippedComposites };
}

function statSafeDir(dir: string): boolean {
  try {
    return statSync(dir).isDirectory();
  } catch {
    return false;
  }
}

export function expectedBaseDir(slug: string, baseRoot = resolveLoraBaseRoot()): string {
  return libraryJoin(baseRoot, slug);
}

export function basenameOf(filePath: string): string {
  return basename(filePath);
}
