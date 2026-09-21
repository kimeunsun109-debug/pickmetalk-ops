# Yuna Identity Workbench Output

| Method | Folder | Role |
|--------|--------|------|
| A | method-a-v8-composite | **Identity PASS** — LOCK 픽셀 합성 |
| B | method-b-mj-cref | MJ 생성 시 --cref face lock |
| C | method-c-prompt-pack | ComfyUI/SDXL 텍스트 identity |
| D | method-d-face-qa | LOCK 대비 유사도 QA |

**권장 운영**: body는 B/C로 생성 → 최종 identity는 A로 LOCK 합성.