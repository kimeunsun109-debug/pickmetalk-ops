# 유나(Yuna) LOCK PRIMARY

## Canonical face (immutable)

| File | Role |
|------|------|
| `01_front_main.jpg` | **LOCK PRIMARY** — 모든 합성·QA·MJ cref 기준 |
| `01_front_main.png` | 원본 고해상도 보관 |

이 파일들은 **수정·재생성하지 않습니다.** 상황만 바꿀 때 body를 따로 만들고 v8 합성으로 얼굴을 LOCK합니다.

## 빠른 실행

```bash
npm run yuna:identity          # 4가지 방법 한번에
npm run face:composite -- --character=yuna --target=path/to/body.jpg --debug
```

## Windows master 경로 (운영)

```
D:\PickMeTalk_PhotoLibrary\master\yuna\01_front_main.jpg
```

환경변수: `FACE_LOCK_YUNA`, `MJ_CREF_YUNA`
