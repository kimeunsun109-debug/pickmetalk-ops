# Face Composite v8 — LOCK PRIMARY 자연 합성

v7에서 **Identity PASS** 후 멈춘 「붙여넣은 티」 문제를 해결하는 v8 파이프라인입니다.

## v7 → v8 변경 요약

| v7 문제 | v8 해결 |
|--------|---------|
| 전체 이미지 워핑 → 머리 왜곡 | **얼굴 패치만** 크롭 후 워핑 |
| 사각 마스크 / 육각 경계 | **타원 마스크** + distance feather + Laplacian blend |
| 이마–헤어라인 수평선 | 이마 fade + **타겟 fringe 머리** 오버레이 |
| 볼·귀 세로 경계 | transition band LAB 색상 매칭 + frequency blend |
| 턱–목 스티커 | jaw-neck 확장 + 색상/조명 전환대 |
| 앞머리 레이어 | hair segmenter + fringe zone 합성 |

**Identity 원칙**: LOCK 픽셀 코어는 유지. GFPGAN/재생성 없음.

## LOCK PRIMARY 경로

캐릭터별 `01_front_main.jpg`:

```
D:\PickMeTalk_PhotoLibrary\master\{character}\01_front_main.jpg
```

환경변수 오버라이드: `FACE_LOCK_YUNA`, `FACE_LOCK_NARIN`, …

## 사용법 (Windows / Cloud)

```bash
# Python 의존성 (1회)
pip install -r scripts/face-composite/requirements.txt

# 합성
npm run face:composite -- --character=yuna --target=D:/path/to/body.jpg --debug

# 직접 지정
npm run face:composite -- \
  --lock=D:/PickMeTalk_PhotoLibrary/master/yuna/01_front_main.jpg \
  --target=D:/path/to/body.jpg \
  --output=D:/out/composite.jpg \
  --color-match=0.7 \
  --hair-strength=0.95
```

출력: `data/face-composite/output/{character}/`  
디버그 마스크: `data/face-composite/debug/` (`--debug`)

## 파라미터 튜닝

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--color-match` | 0.7 | 턱·볼 전환대 LAB 색상 매칭 강도 |
| `--hair-strength` | 0.95 | 타겟 앞머리/옆머리 오버레이 |
| `--no-laplacian` | off | Laplacian pyramid blend 비활성 |
| `--frontal` | auto | 정면: LOCK 픽셀 하드 교체 (`auto`/`on`/`off`) |

### 정면 모드 (`--frontal auto`)

LOCK·타겟 모두 정면이면:
- 회전 최소(±6°), 스케일+이동 정렬
- 얼굴 타원 **내부 LOCK 픽셀 100%** (`paste_frontal_identity`)
- 색상/주파수 블렌드 없음 → identity 동일 우선

## 다음 단계 (로컬)

1. 실제 `01_front_main`으로 테스트 (현재 repo는 dev fallback selfie 사용)
2. 각도 차이 큰 타겟은 Master 16장 중 가장 가까운 각도 body 사용
3. Hybrid Factory ComfyUI 슬롯에 `face:composite` 후처리 연결
