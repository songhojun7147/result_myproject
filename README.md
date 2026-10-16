# 선박·운항 분야의 딥러닝 활용 — 프로젝트 결과물

조선해양 딥러닝 교육(2026.10) 프로젝트로 만든 프로그램 두 개입니다.

| 폴더 | 내용 |
|---|---|
| [`fuel_dashboard/`](fuel_dashboard/) | 경제 운항 계획 — 항로·연료 모드별 항차 비용 비교 (브라우저 앱) |
| [`ship_cam_monitor/`](ship_cam_monitor/) | 위험작업 인식 — CCTV 영상으로 작업 유형·보호구·밀폐구역 진입 감시 (PC 프로그램) |

---

## 1. 경제 운항 계획 (`fuel_dashboard/index.html`)

> ⚠️ **공개용 샘플 데이터입니다.** 운항 기록, 항차, 모델 예측표는 모두 물리식(연료 소모량 ∝ RPM³)으로 만든 가상 데이터입니다. 실제 선박 데이터가 아닙니다. 모델 비교표의 오차 수치만 발표에서 공개한 실제 결과입니다.

**실행:** `index.html`을 브라우저로 열면 됩니다. 설치가 필요 없습니다.

**기능**
- 출발항·도착항을 고르면 파나마 운하 경유와 희망봉 경유의 연료비·통행료·항해일수를 비교합니다.
- LPG 모드와 FO 모드의 하루 소모량을 비교합니다.
- 기상 예보에서 BF 6 이상이면 LPG 대신 FO로 계산합니다.
- ECA 해역 구간은 MGO 사용량을 따로 계산합니다.

**모델 비교 (실제 결과, 검증 300일)**

| 모델 | 일일 연료비 오차 (MAE) |
|---|---|
| 평균 (기준선) | $1,636 |
| 선형회귀 | $1,796 |
| 랜덤포레스트 | $1,669 |
| XGBoost | $1,755 |
| **신경망 MLP (PyTorch)** | **$1,596** |

입력은 M/E RPM, 연료 모드(LPG/FO), 계절입니다. 데이터는 시간 순서대로 나눠 앞 722일로 학습하고 뒤 300일로 검증했습니다.

---

## 2. 위험작업 인식 (`ship_cam_monitor/ship_cam_monitor.py`)

**구성**
1. **객체 탐지·추적** — YOLO11m + BoT-SORT + ReID
2. **작업 유형 판별** — CLIP Zero-shot (추가 학습 없이 설명 문장과 화면의 유사도로 판별)
3. **보호구 판별** — YOLO11 전이학습 + 자세 추정 (안전모·보안경·장갑·하네스)
4. **PERMIT CHECK** — 밀폐구역은 맨홀 자동 탐지 후 진입·체류 시간 감시, IMO MSC.581(110) 기준 가스 측정값 확인

**설치** (`ship_cam_monitor` 폴더에서)
```bash
pip install -r requirements.txt
```

**모델 파일** (용량이 커서 저장소에 포함하지 않았습니다. `ship_cam_monitor.py`와 같은 폴더에 두세요)
- `yolo11m.pt`, `yolo11m-pose.pt` — [Ultralytics](https://docs.ultralytics.com/models/yolo11/)에서 받을 수 있습니다.
- `clip-vit-base-patch32/` — 인터넷이 되는 PC에서 한 번 받아 두면, 그 뒤에는 인터넷 없이 작동합니다.
  ```bash
  python -c "from transformers import CLIPModel, CLIPProcessor as P; n='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(n).save_pretrained('clip-vit-base-patch32'); P.from_pretrained(n).save_pretrained('clip-vit-base-patch32')"
  ```
- `ppe_best.pt` — 보호구 전이학습 모델입니다(포함하지 않음). 없으면 보호구 판정만 꺼지고 나머지는 작동합니다.

**실행**
```bash
python ship_cam_monitor.py
```
[영상 열기] → 작업구역 지정 → [▶ 재생] 순서로 사용합니다.

**주의**
- 공개 영상(Pexels)과 훈련 영상으로만 검증했습니다. 실선 CCTV 환경 검증이 필요합니다.
- 맨홀은 노란 테두리를 기준으로 찾습니다. 다른 색의 맨홀은 감지하지 못합니다.
- 경보는 감시자에게 확인을 요청하는 용도입니다. 최종 안전 판단은 사람이 해야 합니다.
