# 유나 얼굴 일관성 — 4가지 방법

제공된 `01_front_main` 정면 레퍼런스를 기준으로, 같은 얼굴을 유지하는 4가지 경로입니다.

## LOCK PRIMARY

```
reference/yuna/01_front_main.jpg   (repo)
D:\PickMeTalk_PhotoLibrary\master\yuna\01_front_main.jpg   (Windows 운영)
```

## 한번에 실행

```bash
npm run yuna:identity
```

출력: `data/yuna-identity/`

| Method | 폴더 | 역할 | Identity |
|--------|------|------|----------|
| **A** | `method-a-v8-composite/` | LOCK 픽셀 합성 | ✅ 가장 확실 |
| **B** | `method-b-mj-cref/` | MJ `--cref` 명령 | △ 생성 단계 보조 |
| **C** | `method-c-prompt-pack/` | ComfyUI/SDXL 텍스트 | △ 보조 |
| **D** | `method-d-face-qa/` | LOCK 대비 QA | 검증용 |

## 권장 운영 플로우

```
1. body/상황 생성  →  MJ --cref (B) 또는 ComfyUI (C)
2. identity 최종 고정  →  v8 합성 (A)
3. QA  →  face QA (D) + 눈으로 확인
```

**InstantID / inswapper / GFPGAN은 identity drift 때문에 사용하지 않습니다.**

## Method A — v8 pixel LOCK (권장)

```bash
npm run face:composite -- --character=yuna --target=body.jpg --debug
```

- LOCK 코어 픽셀 유지
- 경계·머리·색상만 v8 처리

## Method B — MJ --cref

1. `01_front_main.jpg`를 Discord/CDN에 업로드
2. `.env`: `MJ_CREF_YUNA=https://...`
3. `data/yuna-identity/method-b-mj-cref/YUNA_MJ_CREF_COMMANDS.md` 명령 실행

## Method C — ComfyUI / SDXL

`method-c-prompt-pack/yuna-identity-prompt-pack.json` 사용

- IP-Adapter Face + `01_front_main` reference
- img2img denoise ≤ 0.45 (body만 변경)
- 최종은 Method A로 LOCK 합성

## Method D — Face QA

`method-d-face-qa/qa-report.json` — LOCK 대비 heuristic 유사도

> 현재 QA는 32×32 grayscale heuristic입니다. 운영에서는 InsightFace 업그레이드 권장.
