# Character LoRA (RTX, no face composite)

미드저니로 뽑은 베이스 얼굴을 LoRA로 학습하고, **얼굴 합성 없이** 텍스트만으로 다시 그려 동일 인물인지 확인합니다.

합성·페이스스왑·블렌드는 학습과 샘플링 모두에서 거절합니다. 베이스 사진은 복원 목표로만 쓰고, 확인은 ArcFace 코사인 유사도로만 합니다.

## 베이스 위치

```
D:\PickMeTalk_PhotoLibrary\base\
  yuna\
  narin\
  yunseo\
  eunha\
  jiyu\
```

캐릭터당 정면·좌·우가 섞인 얼굴 사진 5장 이상. 파일명에 `_v8`, `composite`, `swap`이 있거나 sidecar `method`가 합성이면 학습에서 빠집니다.

## Windows RTX

```powershell
pip install -r scripts/lora/requirements.txt
npm run lora:prepare -- --character=yuna
npm run lora:rtx -- --character=yuna
```

`lora:rtx`는 NVIDIA RTX에서만 SDXL LoRA를 학습하고, 프롬프트만으로 샘플 3장을 만든 뒤 베이스와 비교합니다. 평균 유사도 0.40 이상이고 최저 0.32 이상이면 동일 인물입니다.

이 클라우드 환경에는 RTX와 `D:\PickMeTalk_PhotoLibrary\base` 원본이 없으므로, 여기서는 동일 인물이라고 판정하지 않습니다.

## 판정

| 결과 | 의미 |
|------|------|
| `same_person` | ArcFace로 생성 샘플이 베이스와 같은 사람 |
| `different_person` | LoRA 샘플이 다른 사람 |
| `review` | 유사도가 경계 |
| `blocked` / `rtx_required` | GPU·샘플·베이스가 없어 확인 불가 |
| `face_composite_forbidden` | 합성 경로는 점수가 높아도 실패 |

픽셀 프록시 임베딩과, 베이스 파일을 그대로 복사한 샘플은 합격으로 치지 않습니다.
