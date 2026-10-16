"""선박 작업 카메라 모니터 — 사람 추적 · 구역별 작업 판단 · 보호구 착용 판정

실행: python ship_cam_monitor.py
같은 폴더에 yolo11m.pt(사람 추적)가 있어야 합니다. ppe_best.pt(안전모·보안경)가 있으면 착용 판정도 합니다.

사용법
1. [영상 열기]로 영상 선택 → 첫 화면에서 멈춤
2. [＋ 작업구역] → 꼭짓점 클릭, 우클릭(또는 Enter)으로 완료 → 작업 종류 선택
   (CHIPPING / MOORING / ALOFT / HOT WORK / ENCLOSED SPACE / 일반 작업)
   [＋ 위험구역] → 스냅백 구역 등 들어가면 안 되는 곳
3. [▶ 재생]

작업 판단
- 구역에 지정한 작업 + 화면 단서(불꽃, 해치로 사라짐)로 판단
- 불꽃 감지: HOT WORK 구역이 아닌 곳에서 불꽃이 보이면 '무허가 화기작업 의심' 경보
- ENCLOSED SPACE(해치 입구) 구역: 구역 안에서 사람이 사라지면 '탱크 진입'으로 보고 타이머 시작
  · 퇴장은 해치 밖 갑판에서 전신이 보이는 상태로 3초 유지될 때만 인정
  · 체류 제한(기본 30분) 초과 시 경보 → 올라와서 O₂·가스 재측정 후 재진입
  · [가스 측정 기록] 없이(또는 제한 시간보다 오래된 기록으로) 진입하면 경보
  · [이름 지정]으로 탱크 안 인원의 이름을 넣으면 화면과 목록에 이름으로 표시
- 작업구역 안의 사람: 안전모·보안경 미착용이면 경보 (여러 장면의 다수결로 판정)
- 사람이 있는 작업구역 화면을 CLIP(이 PC에 저장된 사진-문장 모델)으로 보고 작업 종류를 추정해 표시하고,
  등록한 작업과 계속 다르면 '작업 불일치' 경보 (MOORING / HOT WORK / CHIPPING / ALOFT / 작업 없음)
"""
import collections
import json
import os
import queue
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO

try:
    import winsound
except ImportError:  # Windows 외 환경
    winsound = None

MODEL_PATH = Path(__file__).with_name("yolo11m.pt")
PPE_PATH = Path(__file__).with_name("ppe_best.pt")
# 화면으로 작업 종류 추정: CLIP(사진-문장 사전학습 모델)에 작업 설명 문장을 주고 가장 비슷한 것을 고름 (추가 학습 없음)
# 시험: 무료 작업 영상 21개 장면 89%, 걷기만 하는 장면은 23/23 '작업 없음' (작업 없음 문장이 없으면 전부 HOT으로 오판)
CLIP_DIR = Path(__file__).with_name("clip-vit-base-patch32")  # 다른 PC로 옮길 때: 프로그램 옆 폴더에 둔 CLIP 사용
CLIP_ID = str(CLIP_DIR) if CLIP_DIR.is_dir() else "openai/clip-vit-base-patch32"  # 이 PC에 저장된 것만 사용 (인터넷 연결 안 함)
WORK_PROMPTS = {
    "ALOFT": ["a worker at height wearing a safety harness", "a rope access worker hanging on a wall",
              "a worker on scaffolding high above the ground", "a worker hanging over the side of a ship hull"],
    "CHIPPING": ["a worker grinding metal with an angle grinder", "metal grinding with sparks",
                 "a worker chipping rust off steel with a hammer", "a worker using a power tool on a steel surface"],
    "HOT WORK": ["a welder welding metal with a welding mask", "arc welding with bright light and smoke",
                 "a person welding steel pipes", "gas cutting metal with a torch"],
    "MOORING": ["sailors handling mooring ropes on a ship deck", "a worker pulling a thick rope at a port",
                "a mooring winch with ropes on a ship", "crew members tying a ship to the dock",
                # 계류에만 있는 모습 추가: 페리 줄작업 영상이 BOLT TIGHTENING 과 반반으로 헷갈려 MOORING 으로 안 바뀌던 문제
                # (페리 8장 중 4→7, 선수갑판 4/5→5/5, 해치커버 BOLT 그대로, 기존 작업 영상 86.3→86.6%)
                "thick mooring ropes coiled on a ship deck next to bollards", "mooring lines tied around bitts on a ship deck",
                "ferry crew in high visibility vests handling heavy ropes at the stern",
                "a sailor throwing a mooring line from a ship to the pier"],
    # 해치커버 닫고 볼트 조이기: 실제 야간 선박 영상 96/100. 조임 공구는 작고 어두워 '공구' 문장은 0% → 자세·장소로 알아봄
    # (기존 작업 영상 정확도 89.2% → 86.3%, 문장 6종 비교 후 선택)
    "BOLT TIGHTENING": ["crew kneeling on a ship deck tightening bolts with a torque wrench",
                        "workers closing a cargo hold hatch cover and tightening the cleats"],
    "NO WORK": ["people walking on a street", "people standing and talking", "an empty ship deck",
              "a person walking on a deck"],
}
# 하네스 착용(ALOFT 구역 사람만): 사람 부분을 잘라 '착용' 문장과 '미착용' 문장 중 어느 쪽에 가까운지 비교
# 시험: 무료 작업 영상 속 사람 337명 AUC 0.97 — 기준 -0.007에서 미착용 90% 잡고 착용자 오경보 9% (한 장 기준, 실제는 다수결)
HARNESS_POS = ["a worker wearing a full body safety harness with straps over the shoulders and around the legs",
               "a worker clipped to a lanyard with a safety harness", "a rope access worker in a harness"]
HARNESS_NEG = ["a worker in normal work clothes without any harness", "a worker in a jacket and trousers",
               "a person in a t-shirt", "a worker in a high visibility reflective vest without a harness",
               "a sailor wearing a life jacket", "a welder in protective clothing"]
HARNESS_TH = -0.007
# 안전모: 보호구 모델이 안전모/맨머리를 못 찾았을 때만 CLIP에 다시 물음 (야간·웅크린 자세에서 보호구 모델이 거의 못 찾음)
# 시험: 주간 정답 206명 AUC 0.93 — 기준 -0.01에서 미착용 78% 잡고 오경보 8% / 실제 야간 해치커버 영상 71명(전원 착용) 96% '착용'
HELMET_POS = ["a worker wearing a hard hat", "a person wearing a white safety helmet",
              "a worker with a construction helmet on his head"]
HELMET_NEG = ["a worker with no hat", "a person with bare head", "a worker wearing a knit beanie",
              "a person wearing a baseball cap", "a worker wearing a hood"]
HELMET_TH = -0.01
# BA(공기호흡기) 착용 = 보호구. 전면 마스크가 눈을 덮으므로 BA 착용자는 보안경 경고 안 함 (사용자 지시)
# 시험: 밀폐구역 구조훈련 영상 사람 72명 — 공기통 메고 마스크 쓴 사람 대부분 +, 일반 작업복 대부분 − (다수결로 사용)
BA_POS = ["a firefighter wearing a self-contained breathing apparatus with an air cylinder on the back",
          "a person wearing a full face breathing mask connected to an air tank",
          "a rescuer with a compressed air cylinder backpack"]
BA_MIN_CONF_HATCH = 0.4  # 맨홀 근처 사람(진입 후보)은 해치·보조자에 가려 확신도가 낮음 (C.mp4 진입자 평균 0.61)
BA_WIN, BA_MIN_CONF = 4, 0.6  #BA 판정 창(최근 판정 수), BA 판정에 쓰는 최소 사람 확신도 (비트 뒤에 가려진 사람이 비트를 공기통으로 보여 BA 오판)
BA_TH = 0.020  # C.mp4 11~15초: BA 없는 사람 최고 +0.014, 마스크 쓴 사람 대부분 +0.025 이상. 0.020 넘는 장면 2번이면 BA 확정 # 공기통·마스크 문장이 이만큼 더 가까워야 1표 (해치커버 영상에서 BA 없는 사람을 BA로 오판했었음)
BA_NEG = ["a worker in coveralls without any breathing apparatus", "a worker wearing a safety helmet and coveralls",
          "a person holding a tablet", "a sailor in uniform"]
# 머리 기준 판정: 자세 추정(코·눈·귀)으로 머리를 찾아 확대 → 보호구 모델, 못 찾으면 '밝은 헬멧 색' 비율
# 실제 야간 해치커버 영상: 일반 모자 0.00~0.02, 흰 헬멧 0.04~0.28 (사람 상자 전체로 보면 노란 설비를 헬멧으로 착각했음)
POSE_PATH = Path(__file__).with_name("yolo11m-pose.pt")
HELMET_BRIGHT_Y, HELMET_BRIGHT_N = 0.06, 0.03  # 이상이면 착용, 이하면 미착용, 사이는 보류
HAIR_DARK_N, HAIR_MIN_PX = 0.5, 30  # 머리 윗부분이 절반 이상 검고 헬멧 색이 없으면 맨머리(미착용). 머리 30px 미만은 판정 안 함
HEAD_SEC = 0.5          # 사람마다 0.5초에 1번 머리 판정
REID_SEC, REID_SIM, REID_MARGIN = 300, 0.90, 0.03  # 사라진 지 5분 안에, 겉모습(옷차림) 유사도 0.88 이상이면 같은 사람 → 같은 이름
HARNESS_SEC = 1.0       # 사람마다 1초에 1번 판정
WORK_CLS_SEC = 1.0      # 구역마다 1초에 1번 추정
WORK_MISMATCH = 0.7     # 등록 작업과 다른 작업이 (누적 평균) 70% 이상이면 '작업 불일치' 경보
FIRE_CLIP_MIN = 0.3     # 불꽃 감지는 CLIP의 HOT WORK 확률이 30% 이상일 때만 인정 (조명 반사 오경보 방지)
IMGSZ = 1280  # 960 → 1280: 크레인 기둥 고소작업 영상에서 사람을 못 찾는 장면 23/40 → 8/40 (속도는 34 → 36 ms)
MAX_SIDE = 1920  # 이보다 큰 영상(4K 등)은 줄여서 분석
DET_CONF = 0.1  # 추적기에는 낮은 신뢰도 탐지까지 넘겨야 가려짐 후에도 같은 사람으로 이어짐 (새 ID 시작 기준은 추적기가 0.25로 따로 적용)
CONFIRM_N = 5   # 새 사람은 5번 연속 추적된 뒤에 번호 부여 (순간적인 오탐지에 번호가 낭비되지 않게)
RELINK_SEC = 3.0  # 추적이 끊긴 뒤 3초 안에 근처에서 다시 나타나면 원래 번호를 이어 붙임

# BoT-SORT + ReID(외형 특징): 가려졌다 다시 나타난 사람을 같은 사람으로 인식. 설정 파일은 임시 폴더에 생성
TRACKER_YAML = Path(tempfile.gettempdir()) / "ship_cam_botsort.yaml"
TRACKER_YAML.write_text("""tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 90
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: True
model: auto
""", encoding="utf-8")
ENTER_N, EXIT_N = 3, 8  # 분석 프레임 기준: 3번 연속 안에 있으면 '진입', 8번 연속 없으면 '이탈' (깜빡임 방지)
WORK_GRACE_SEC = 60  # 작업구역에서 사라져도 60초 안에 돌아오면(또는 근처에서 새 번호로 나타나면) 작업 시간을 이어서 셈
WORK_TYPES = ["CHIPPING", "MOORING", "ALOFT", "HOT WORK", "ENCLOSED SPACE", "BOLT TIGHTENING", "GENERAL WORK"]
ENCLOSED = "ENCLOSED SPACE"
FULL_OUT_SEC = 3.0  # 해치에서 나온 사람이 '완전히 나옴'으로 인정되려면 전신이 보이는 채로 3초 유지
# 진입 전 가스 기준: IMO MSC.581(110) 7.4 (2025 채택, A.1050(27) 대체) — 안정된 측정값이 모두 만족해야 진입
# 진입 절차 확인표: IMO MSC.581(110) 부록 '밀폐구역 진입 허가서' 1장(진입 전 준비)을 CCTV 감시자가 재확인할 항목으로 정리
ENTRY_CHECKS = [
    "Entry permit issued and signed by the master / responsible person",
    "Space thoroughly ventilated, and ventilation kept running while occupied",
    "Space isolated — pipelines / valves blanked, power isolated",
    "GAS",  # 가스 측정: 마지막 기록을 자동으로 보여 줌
    "No residual gas or cargo vapour — tested at different levels",
    "Attendant (watchman) in constant attendance at the entrance",
    "Officer of the watch (bridge / ECR / CCR) advised of the entry",
    "Communication tested and emergency signals agreed",
    "Rescue & resuscitation equipment at the entrance (tripod, SCBA, lifeline, stretcher)",
    "Entrants carry personal gas detectors (O2, CO2, flammable, toxic)",
    "Access and lighting adequate",
]
CHECK_DEADLINE_SEC = 60
# 움직이는 카메라(휴대폰 등)는 해치 구역이 맞지 않음 → 화면 가장자리가 아닌 곳에서 2초 넘게 사라진 사람은 '진입 의심'으로 감시자에게 물어봄
POSSIBLE_ENTRY_SEC, EDGE_MARGIN = 2.0, 0.08
# 맨홀 찾기: 노란 테두리 + 안쪽에 노란색 없음(구멍) → 타원. 손으로 든 카메라(C.mp4) 33~39초 매초 찾음, 맨홀 없는 장면은 '없음'
MANHOLE_KEEP_SEC = 2.0   # 마지막으로 찾은 맨홀을 2초 동안 유지  # 확인표가 뜬 뒤 60초 안에 감시자가 확인하지 않으면 경보
GAS_FIELDS = [("O2", "O₂ (%vol)", "20.9"), ("CO2", "CO₂ (ppm)", "400"), ("LFL", "Flammable gas (%LFL)", "0"),
              ("H2S", "H₂S (ppm)", "0"), ("CO", "CO (ppm)", "0")]
O2_OK = (20.8, 21.0)    # IMO 20.9 %vol — 측정기 오차 ±0.1 허용
CO2_MAX_PPM = 5000      # 0.5 %vol 미만
LFL_MAX = 1.0           # 1 %LFL 미만
OEL_PPM = {"H2S": 1.0, "CO": 25.0}  # 노출기준 예시(ACGIH TLV-TWA). IMO는 '국가·국제 기준'을 따르라고 함 → 기국·회사 기준으로 교체. 50% 미만이어야 함
PPE_KO = {"helmet": "Helmet", "goggles": "Goggles", "harness": "Harness", "ba": "BA set", "gloves": "Gloves"}
# 화면 경고 문구 (ANSI Z535 관례: DANGER = 즉시 사망·중상 위험, WARNING = 중상 가능)
PPE_WARN = {"ba": "", "gloves": "WARNING: NO GLOVES", "helmet": "WARNING: NO HELMET", "goggles": "WARNING: NO SAFETY GOGGLES", "harness": "WARNING: NO SAFETY HARNESS"}
DANGER_ZONE_TXT = "DANGER: RESTRICTED ZONE"
GOGGLE_MISSING_SEC = 10
# 장갑(ppe v3): 시험 사진에서 '장갑 착용' mAP50 0.72, '맨손'은 0.04(거의 못 잡음) → 착용 확인만 믿고, MOORING 작업에서 10초간 확인 안 되면 경고
GLOVE_MISSING_SEC = 10
GLOVE_WORKS = ("MOORING",)  # 장갑 필수 작업 (사용자 지시)  # 화면에 나타난 뒤(또는 마지막으로 보안경이 보인 뒤) 10초 동안 보안경이 안 보이면 미착용
PPE_CLASSES = {"helmet": ("helmet", "no_helmet"), "goggles": ("goggles", "no_goggle")}  # (착용, 미착용)
PPE_HIST = 9  # 최근 9번 판정의 다수결
HELMET_ABSENT_MIN_H = 160  # 사람 키가 이 픽셀 이상인데 안전모가 안 보이면 미착용으로 셈 (멀리 작은 사람은 판정 보류)
ZONE_COLOR = {"work": (60, 150, 60), "danger": (40, 40, 210), "enclosed": (170, 90, 20)}  # BGR
# 화면 디자인: 산업용 CCTV 관제 화면 스타일 (주간 모드). 영어는 Bahnschrift(DIN 계열), 한글이 섞이면 맑은 고딕
UI = {"bg": "#eef1f5", "panel": "#ffffff", "line": "#d5dbe3", "ink": "#1c2333", "muted": "#6b7686",
      "navy": "#1f3a5f", "red": "#c62828", "green": "#2e7d4f"}
UI_FONT, NUM_FONT, MONO_FONT = "Segoe UI", "Bahnschrift", "Consolas"
BGR = {"track": (0, 190, 255), "ok": (80, 160, 46), "bad": (50, 50, 214), "navy": (95, 58, 31)}  # 영상 위 선 색


def _font(file, size, bold=False):
    try:
        f = ImageFont.truetype(f"C:/Windows/Fonts/{file}", size)
        if bold:
            try:
                f.set_variation_by_name("Bold")
            except Exception:
                pass
        return f
    except OSError:
        return None


F_EN = _font("bahnschrift.ttf", 14) or ImageFont.load_default()
F_EN_B = _font("bahnschrift.ttf", 14, bold=True) or F_EN
F_KO = _font("malgunbd.ttf", 14) or F_EN
FONT_B = F_KO  # 예전 코드 호환


def font_for(text, bold=False):
    return F_KO if any("\uac00" <= ch <= "\ud7a3" for ch in text) else (F_EN_B if bold else F_EN)


def mmss(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def pname(pid):
    """표시 번호 → '사람 A', '사람 B', … '사람 Z', '사람 AA'"""
    s, n = "", pid
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return f"Person {s}"


def ago(sec):
    return f"{int(sec // 60)} min {int(sec % 60):02d} s" if sec >= 60 else f"{int(sec)} s"


def find_manhole(frame):
    """노란 테두리 맨홀 → ((cx, cy), (a, b), angle) 또는 None. 계산은 폭 640으로 줄여서"""
    H0, W0 = frame.shape[:2]
    k = 640 / W0 if W0 > 640 else 1.0
    f = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else frame
    H, W = f.shape[:2]
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 110, 120), (38, 255, 255))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        if cv2.contourArea(c) < H * W * 0.01 or len(c) < 20:
            continue
        (cx, cy), (a, b), ang = cv2.fitEllipse(c)
        if min(a, b) < 20 or max(a, b) > 1.6 * W or min(a, b) / max(a, b) < 0.25:
            continue
        inner = np.zeros((H, W), np.uint8)
        cv2.ellipse(inner, ((cx, cy), (a * 0.55, b * 0.55), ang), 255, -1)
        ring = np.zeros((H, W), np.uint8)
        cv2.ellipse(ring, ((cx, cy), (a, b), ang), 255, max(3, int(min(a, b) * 0.08)))
        inside_yellow = (yellow[inner > 0] > 0).mean() if inner.any() else 1
        ring_yellow = (yellow[ring > 0] > 0).mean() if ring.any() else 0
        if ring_yellow > 0.40 and inside_yellow < 0.15 and (best is None or ring_yellow > best[0]):
            best = (ring_yellow, ((cx / k, cy / k), (a / k, b / k), ang))
    return best[1] if best else None


def in_ellipse(p, e, grow=1.0):
    (cx, cy), (a, b), ang = e
    t = np.deg2rad(ang)
    dx, dy = p[0] - cx, p[1] - cy
    x = dx * np.cos(t) + dy * np.sin(t)
    y = -dx * np.sin(t) + dy * np.cos(t)
    return (x / (a / 2 * grow)) ** 2 + (y / (b / 2 * grow)) ** 2 <= 1


def head_box(box, kp=None, kc=None):
    """머리 영역 (헬멧이 있는 정수리 위까지). 자세 추정 점(코·눈·귀)이 있으면 그 기준, 없으면 서 있는 사람만 상자 위쪽"""
    x1, y1, x2, y2 = box
    h = y2 - y1
    if kp is not None and kc is not None:
        pts = np.array([kp[i] for i in range(5) if kc[i] > 0.3])
        if len(pts):
            cx, cy = pts.mean(0)
            sh = [kp[i] for i in (5, 6) if kc[i] > 0.3]
            sc = np.linalg.norm(sh[0] - sh[1]) * 0.45 if len(sh) == 2 else (np.ptp(pts[:, 0]) + 1) * 1.2 if len(pts) > 1 else h * 0.12
            sc = max(sc, h * 0.08, 14)
            return int(cx - 1.3 * sc), int(cy - 1.7 * sc), int(cx + 1.3 * sc), int(cy + 1.0 * sc)
        return None  # 머리가 안 보임 (뒤돌아 엎드림 등) → 판정 안 함
    if h >= 1.3 * (x2 - x1):
        return int(x1), int(y1 - 0.05 * h), int(x2), int(y1 + 0.25 * h)
    return None


def crown_box(box, kp, kc):
    """정수리(헬멧 자리): 어깨 가운데 → 얼굴 방향으로 더 나간 곳. 고개를 숙이거나 엎드려도 머리 방향을 따라감
    (예전: '얼굴보다 화면 위쪽'으로 가정 → 엎드린 사람은 얼굴·몸을 보고 '흰색 없음'이라 흰 헬멧 착용자에게 빨간 H 오경보)
    얼굴(코·눈)이 2점 이상, 양 어깨가 모두 확실히 보일 때만. 아니면 판정 보류"""
    if kp is None or kc is None:
        return None
    face = np.array([kp[i] for i in (0, 1, 2) if kc[i] > 0.5])
    if len(face) < 2 or kc[5] < 0.5 or kc[6] < 0.5:
        return None
    f = face.mean(0)
    sh = (kp[5] + kp[6]) / 2
    v = f - sh                      # 목 → 얼굴 방향
    n = np.linalg.norm(v)
    if n < 5:
        return None
    sc = max(np.linalg.norm(kp[5] - kp[6]) * 0.35, 10)  # 머리 크기 ≈ 어깨 폭의 1/3
    c = f + v / n * sc * 0.9        # 얼굴에서 머리 방향으로 조금 더 → 정수리
    return int(c[0] - sc), int(c[1] - sc), int(c[0] + sc), int(c[1] + sc)


def helmet_brightness(head):
    """정수리 조각에서 밝은 헬멧 색(흰색, 또는 노랑·주황) 비율"""
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (vv > 140) & (ss < 90)
    yellow = (vv > 140) & (ss >= 90) & (hh >= 8) & (hh <= 35)
    return float((white | yellow).mean())


def hair_dark(frame, hb):
    """머리 상자 윗부분 가운데 띠(옆 배경·아래 얼굴 제외)에서 검은 머리 비율과 헬멧 색 비율
    시험: 계류 영상 맨머리 대부분 0.5↑ / 헬멧 착용(페리·구조훈련·해치커버) 전원 0.5 미만 또는 헬멧 색 있음"""
    x1, y1, x2, y2 = hb
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    band = frame[max(0, int(y1 + h * 0.15)):min(H, int(y1 + h * 0.5)), max(0, int(x1 + w * 0.25)):min(W, int(x2 - w * 0.25))]
    if band.size == 0:
        return 0.0, 1.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] < 80) & (hsv[..., 1] < 150)).mean()), helmet_brightness(band)


def corners(img, p1, p2, col, t=2):
    """사람 표시: 상자 대신 네 모서리 괄호 (CCTV 관제 화면 방식)"""
    (x1, y1), (x2, y2) = p1, p2
    L = max(8, min(x2 - x1, y2 - y1) // 5)
    for (x, y, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * L, y), col, t, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * L), col, t, cv2.LINE_AA)


def dashed_poly(img, pts, col, t=2, dash=12, gap=8):
    """위험구역: 점선 테두리"""
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        a, b = a.astype(float), b.astype(float)
        n = max(1, int(np.linalg.norm(b - a) // (dash + gap)))
        for k in range(n + 1):
            s0 = k * (dash + gap) / max(1e-6, np.linalg.norm(b - a))
            s1 = min(1.0, s0 + dash / max(1e-6, np.linalg.norm(b - a)))
            if s0 >= 1:
                break
            cv2.line(img, tuple((a + (b - a) * s0).astype(int)), tuple((a + (b - a) * s1).astype(int)), col, t, cv2.LINE_AA)


def gas_failures(v):
    """IMO MSC.581(110) 7.4 기준으로 안 맞는 항목 목록 (빈 목록이면 진입 가능)"""
    bad = []
    if not O2_OK[0] <= v["O2"] <= O2_OK[1]:
        bad.append(f"O₂ {v['O2']}% (req. 20.9%)")
    if v["CO2"] >= CO2_MAX_PPM:
        bad.append(f"CO₂ {v['CO2']:g} ppm (req. < 5,000)")
    if v["LFL"] >= LFL_MAX:
        bad.append(f"Flammable {v['LFL']:g} %LFL (req. < 1)")
    for k, oel in OEL_PPM.items():
        if v[k] >= oel * 0.5:
            bad.append(f"{k} {v[k]:g} ppm (req. < {oel * 0.5:g} = 50% of OEL)")
    return bad


class Zone:
    def __init__(self, name, kind, pts, work=None):
        self.name, self.kind, self.pts = name, kind, [tuple(map(int, p)) for p in pts]
        self.work = {"일반 작업": "GENERAL WORK"}.get(work, work or "GENERAL WORK") if kind == "work" else None  # 예전 저장 파일 호환
        self.poly = np.array(self.pts, np.int32)
        self.gas_t, self.gas_vals, self.last_exit_t = None, None, None  # 마지막 '기준 통과' 측정 시각·값, 마지막 퇴장 시각
        self.reset()

    def reset(self):
        self.inside, self.hits, self.miss, self.enter_t = set(), {}, {}, {}
        self.away, self.last_feet = {}, {}  # 잠깐 사라진 사람: pid -> (작업 시작 시각, 사라진 시각, 마지막 발 위치)
        self.tank, self.exit_cand = [], {}  # ENCLOSED: 탱크 안 인원 [{pid, name, t_in, warn, over}]
        self.hot, self.hot_on, self.prev_mask = 0.0, False, None
        self.cls_prob, self.cls_t, self.mismatch_alarmed, self.cls_n = None, -1e9, False, 0
        self.check_t, self.check_win = None, None  # 진입 절차 확인 시각, 열려 있는 확인표 창  # 카메라 작업 추정 (확률 누적 평균)

    def contains(self, p):
        return p is not None and cv2.pointPolygonTest(self.poly, (float(p[0]), float(p[1])), False) >= 0

    def color(self):
        return ZONE_COLOR["enclosed" if self.work == ENCLOSED else self.kind]

    def shown(self):
        """화면에 보일 작업 이름 — 맨홀이 발견된 화면의 자동 전체 구역은 ENCLOSED SPACE 로 표시 (판단 기준은 그대로)"""
        return getattr(self, "shown_work", None) or self.work

    def label(self):
        return f"DANGER ZONE · {self.name}" if self.kind == "danger" else f"{self.shown()} · {self.name}"


class WorkCLIP:
    """작업 구역 화면 → 작업 종류 확률. 문장(WORK_PROMPTS) 임베딩은 처음에 한 번만 계산"""

    def __init__(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_ID, local_files_only=True).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID, local_files_only=True)
        self.names = list(WORK_PROMPTS)
        with torch.no_grad():
            rows = []
            for k in self.names:
                x = self.proc(text=WORK_PROMPTS[k], return_tensors="pt", padding=True).to(self.dev)
                f = self.unit(self.model.get_text_features(**x)).mean(0)
                rows.append(f / f.norm())
            self.text = torch.stack(rows)
            self.pairs = {"harness": torch.stack([self.embed_text(HARNESS_POS), self.embed_text(HARNESS_NEG)]),
                          "ba": torch.stack([self.embed_text(BA_POS), self.embed_text(BA_NEG)]),
                          "helmet": torch.stack([self.embed_text(HELMET_POS), self.embed_text(HELMET_NEG)])}

    def embed_text(self, texts):
        x = self.proc(text=texts, return_tensors="pt", padding=True).to(self.dev)
        f = self.unit(self.model.get_text_features(**x)).mean(0)
        return f / f.norm()

    def wear_scores(self, crops_bgr, item):
        """사람 조각마다 (착용 문장 유사도 - 미착용 문장 유사도). item = 'harness' 또는 'helmet'"""
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        with self.torch.no_grad():
            x = self.proc(images=imgs, return_tensors="pt").to(self.dev)
            s = self.unit(self.model.get_image_features(**x)) @ self.pairs[item].T
            return (s[:, 0] - s[:, 1]).cpu().numpy()

    def unit(self, out):
        f = out if self.torch.is_tensor(out) else out.pooler_output  # transformers 버전에 따라 반환형이 다름
        return f / f.norm(dim=-1, keepdim=True)

    def embed_crops(self, crops_bgr):
        """사람 조각 → 겉모습 특징 (같은 사람 다시 알아보기·하네스 판정에 같이 씀)"""
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        with self.torch.no_grad():
            x = self.proc(images=imgs, return_tensors="pt").to(self.dev)
            return self.unit(self.model.get_image_features(**x))

    def probs(self, crops_bgr):
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        with self.torch.no_grad():
            x = self.proc(images=imgs, return_tensors="pt").to(self.dev)
            f = self.unit(self.model.get_image_features(**x))
            return self.torch.softmax(100 * f @ self.text.T, dim=1).cpu().numpy()


class Relinker:
    """추적기 ID → 화면 이름(Person A, B …). 확정 대기 + 끊긴 ID 재연결: 근처에서 3초 안 / 같은 옷차림이면 5분 안"""

    def __init__(self):
        self.map, self.pending, self.last, self.next = {}, {}, {}, 1
        self.seen = {}  # 표시 번호 -> 마지막으로 보인 시각

    def update(self, dets, t, reid=None):
        out, active, shown = [], {self.map[tid] for tid, _, _ in dets if tid in self.map}, set()
        for tid, box, extra in dets:
            x1, y1, x2, y2 = box
            cx, cy, h = (x1 + x2) / 2, y2, max(1, y2 - y1)
            if tid not in self.map:
                # 새 대상은 '사람다울 때'만 이름 부여·이어 붙이기: 확신도 0.5↑ + 관절 6개↑, 또는 확신도 0.7↑
                # (계류 영상: 볼라드(BITT)가 확신도 0.52·관절 4개로 사람 'D'가 됐었음. 실제 사람은 0.72↑ 또는 관절 9개↑)
                kc, cf = (extra[1], extra[2]) if extra and len(extra) >= 3 else (None, 1.0)
                good = cf >= 0.7 or (cf >= 0.5 and (kc is None or int((kc > 0.5).sum()) >= 6))
                if not good:
                    continue
                d = self.match(cx, cy, h, t, active)
                if d is None:
                    self.pending[tid] = self.pending.get(tid, 0) + 1
                    if self.pending[tid] < CONFIRM_N:
                        continue
                    cands = [k for k, lt in self.seen.items() if k not in active and t - lt <= REID_SEC]
                    d = reid(box, cands) if reid and cands else None  # 지금 안 보이는 사람 중 옷차림이 같은 사람
                    if d is None:
                        d, self.next = self.next, self.next + 1
                for old in [k for k, v in self.map.items() if v == d]:  # 이 이름을 쓰던 예전 추적 번호는 끊음
                    self.map.pop(old)                                   # (다시 나타나면 새 사람으로 확인 → 같은 이름이 두 명에게 붙지 않게)
                self.map[tid] = d
                self.pending.pop(tid, None)
                active.add(d)
            d = self.map[tid]
            if d in shown:  # 안전장치: 한 장면에 같은 이름은 한 명만
                continue
            shown.add(d)
            self.last[d] = (t, cx, cy, h)
            self.seen[d] = t
            out.append((d, box, extra))
        live = {tid for tid, _, _ in dets}
        for tid in [k for k in self.pending if k not in live]:
            self.pending.pop(tid)
        for d in [k for k, v in self.last.items() if t - v[0] > RELINK_SEC * 2]:
            self.last.pop(d)
        return out

    def match(self, cx, cy, h, t, active):
        # 최근에 사라진 번호 중, 사라진 위치 근처(시간이 지날수록 허용 거리 증가)·비슷한 키의 가장 가까운 것
        best, bd = None, None
        for d, (lt, lx, ly, lh) in self.last.items():
            gap = t - lt
            if d in active or gap <= 0 or gap > RELINK_SEC or not 0.5 < h / lh < 2:
                continue
            dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if dist <= (h + lh) / 2 * (1.0 + 1.5 * gap) and (bd is None or dist < bd):
                best, bd = d, dist
        return best


class App:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.events = queue.Queue()
        self.zones, self.latest, self.new = [], None, False
        self.path, self.gen, self.paused, self.ended = None, 0, True, False
        self.draw_kind, self.draw_pts = None, []
        self.view_tf = (1.0, 0, 0)
        self.alarm_on, self.flash = False, False
        self.limit_min = 30
        self.wclip = None  # 작업 추정 모델 (check_model 에서 불러옴)
        self.clip_done = False  # CLIP 불러오기 시도가 끝났는지 (성공·실패 무관)
        self.all_frames_on = False  # 모든 장면 분석 모드 (건너뛰지 않음 → 같은 영상이면 매번 같은 결과)
        self.reset_state()
        self.build_ui()
        self.set_status("Loading models...")
        threading.Thread(target=self.check_model, daemon=True).start()
        self.poll()

    def reset_state(self):
        self.first_seen, self.ppe_hist, self.ppe_alarmed, self.wear_t, self.helmet_unknown = {}, {}, set(), {}, set()
        self.first_t = {}  # 사람별 처음 보인 시각 (작업 경과 표시용)
        self.goggle_seen, self.cur_t, self.ppe_on = {}, 0.0, PPE_PATH.exists()  # 보안경 마지막 확인 시각, 현재 영상 시각
        self.app_emb, self.head_t, self.alerts, self.ba_hist, self.ba_sure = {}, {}, [], {}, set()
        self.det_cf = {}  # 이번 장면 사람별 탐지 확신도
        self.last_pos = {}  # 사람별 마지막으로 보인 시각·상자·화면 크기 (진입 의심 판단용)
        self.glove_t, self.glove_seen, self.glove_since = {}, {}, {}  # 장갑 판정 시각, 마지막 확인 시각, 장갑 필수 구역에 들어온 시각
        self.manhole, self.manhole_t, self.on_hatch, self.near_hatch = None, -1e9, {}, {}
        self.manhole_from = 1e9
        self.last_auto_entry = -1e9  # 마지막 자동 진입 시각 (중복 방지)  # 찾은 맨홀, 찾은 시각, 사람별 맨홀 위에 있던 시각  # 사람별 겉모습 특징, 마지막 머리 판정 시각, 현재 경고 목록

    # ---------------- 화면 구성 ----------------
    def build_ui(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", font=(UI_FONT, 10), background=UI["panel"], foreground=UI["ink"])
        st.configure("TCombobox", padding=3)
        self.root.configure(bg=UI["bg"])

        def b(parent, text, cmd, primary=False, **kw):  # 평평한 버튼 + 얇은 테두리
            return tk.Button(parent, text=text, command=cmd, font=(UI_FONT, 9, "bold"), relief="flat", bd=0,
                             padx=12, pady=5, cursor="hand2", highlightthickness=1,
                             bg=UI["navy"] if primary else "#f4f6f9", fg="white" if primary else UI["ink"],
                             activebackground="#2b4c78" if primary else "#e6eaf0",
                             activeforeground="white" if primary else UI["ink"],
                             highlightbackground=UI["navy"] if primary else UI["line"], **kw)

        def sep(parent, bg):
            tk.Frame(parent, width=1, bg=UI["line"]).pack(side="left", fill="y", padx=10, pady=6)

        # 상단: 제목 · 영상 열기 · 재생 · SETUP (평소엔 이것만)
        bar = tk.Frame(self.root, bg=UI["panel"], highlightbackground=UI["line"], highlightthickness=1)
        bar.pack(fill="x")
        tk.Label(bar, text="SHIP WORK MONITOR", bg=UI["panel"], fg=UI["navy"], font=(NUM_FONT, 13, "bold")).pack(
            side="left", padx=(14, 6), pady=9)
        sep(bar, UI["panel"])
        b(bar, "OPEN VIDEO", self.open_video).pack(side="left", padx=3)
        self.play_btn = b(bar, "PLAY", self.toggle_play, primary=True, width=8)
        self.play_btn.pack(side="left", padx=3)
        self.all_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="ALL FRAMES", variable=self.all_var, bg=UI["panel"], activebackground=UI["panel"],
                       font=(UI_FONT, 8, "bold"), fg=UI["muted"],
                       command=lambda: setattr(self, "all_frames_on", self.all_var.get())).pack(side="left", padx=(10, 0))
        self.setup_btn = b(bar, "SETUP", self.toggle_setup)
        self.setup_btn.pack(side="right", padx=14)
        small = lambda text, cmd: tk.Button(bar, text=text, command=cmd, font=(UI_FONT, 8, "bold"), relief="flat", bd=0,
                                            padx=7, pady=2, cursor="hand2", bg="#f4f6f9", fg=UI["ink"],
                                            highlightthickness=1, highlightbackground=UI["line"])
        small("EXIT", self.manual_exit).pack(side="right", padx=2)   # 작은 수동 버튼: 탱크 진입·퇴장 직접 입력
        small("ENTRY", self.manual_entry).pack(side="right", padx=2)
        tk.Label(bar, text="TANK", bg=UI["panel"], fg=UI["muted"], font=(UI_FONT, 8, "bold")).pack(side="right", padx=(0, 4))

        # 설정 줄 (SETUP 누를 때만): 구역 그리기·지우기·저장, 밀폐구역 제한 시간, 탱크 인원·가스 기록
        self.setup_bar = tk.Frame(self.root, bg="#f4f6f9", highlightbackground=UI["line"], highlightthickness=1)
        sb = self.setup_bar
        b(sb, "+ WORK ZONE", lambda: self.start_draw("work")).pack(side="left", padx=(14, 3), pady=6)
        b(sb, "+ DANGER ZONE", lambda: self.start_draw("danger")).pack(side="left", padx=3)
        sep(sb, "#f4f6f9")
        tk.Label(sb, text="ZONE", bg="#f4f6f9", fg=UI["muted"], font=(UI_FONT, 8, "bold")).pack(side="left")
        self.zone_pick = ttk.Combobox(sb, state="readonly", width=22, font=(UI_FONT, 9))
        self.zone_pick.pack(side="left", padx=6)
        b(sb, "DELETE", self.delete_zone).pack(side="left", padx=3)
        b(sb, "NAME TANK ENTRANTS", self.name_tank).pack(side="left", padx=3)
        b(sb, "RECORD GAS TEST", self.record_gas).pack(side="left", padx=3)
        sep(sb, "#f4f6f9")
        b(sb, "SAVE ZONES", self.save_zones).pack(side="left", padx=3)
        b(sb, "LOAD ZONES", self.load_zones).pack(side="left", padx=3)
        self.limit_var = tk.IntVar(value=self.limit_min)
        tk.Spinbox(sb, from_=1, to=120, width=4, textvariable=self.limit_var, font=(NUM_FONT, 11), relief="flat",
                   highlightthickness=1, highlightbackground=UI["line"], command=self.on_limit).pack(side="right", padx=(4, 14))
        tk.Label(sb, text="TANK TIME LIMIT (MIN)", bg="#f4f6f9", fg=UI["muted"], font=(UI_FONT, 8, "bold")).pack(side="right")
        self.limit_var.trace_add("write", lambda *_: self.on_limit())
        self.setup_hint = tk.Label(self.root, bg="#fff8e1", fg="#7a5a00", font=(UI_FONT, 9), anchor="w",
                                   text="SETUP — draw a zone: click points on the video, right-click or Enter to finish, Esc to cancel. "
                                        "Pick a zone in ZONE to delete it, name tank entrants or record a gas test.")

        self.alarm_bar = tk.Frame(self.root, bg=UI["red"])
        tk.Label(self.alarm_bar, text="ALARM", bg="white", fg=UI["red"], font=(NUM_FONT, 11, "bold"), padx=8).pack(
            side="left", padx=(14, 10), pady=7)
        self.alarm_lbl = tk.Label(self.alarm_bar, text="", bg=UI["red"], fg="white", font=(NUM_FONT, 13, "bold"))
        self.alarm_lbl.pack(side="left", pady=6)
        tk.Button(self.alarm_bar, text="ACKNOWLEDGE", command=self.ack_alarm, font=(UI_FONT, 9, "bold"), relief="flat",
                  bg="white", fg=UI["red"], padx=14, pady=4, cursor="hand2").pack(side="right", padx=14, pady=6)

        self.body = body = tk.Frame(self.root, bg=UI["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=10)
        side = tk.Frame(body, bg=UI["panel"], width=330, highlightbackground=UI["line"], highlightthickness=1)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        self.view = tk.Label(body, bg="#dfe3e9", text="OPEN A VIDEO TO START", font=(NUM_FONT, 13), fg=UI["muted"],
                             highlightbackground=UI["line"], highlightthickness=1)
        self.view.pack(side="left", fill="both", expand=True)
        self.view.bind("<Configure>", lambda _: self.render())
        self.view.bind("<Button-1>", self.on_click)
        self.view.bind("<Button-3>", lambda _: self.finish_draw())
        self.root.bind("<Return>", lambda _: self.finish_draw())
        self.root.bind("<Escape>", lambda _: self.cancel_draw())
        self.root.bind("<space>", lambda _: self.toggle_play())

        def head(text):  # 구획 제목: 작은 대문자, 회색 + 아래 얇은 선
            f = tk.Frame(side, bg=UI["panel"])
            f.pack(fill="x", padx=16, pady=(16, 6))
            tk.Label(f, text=text, bg=UI["panel"], fg=UI["muted"], font=(UI_FONT, 8, "bold")).pack(side="left")
            return f

        h = head("ON SCREEN")
        self.total_lbl = tk.Label(h, text="0", bg=UI["panel"], fg=UI["ink"], font=(NUM_FONT, 12, "bold"))
        self.total_lbl.pack(side="right")
        self.ppe_lbl = tk.Label(side, text="", bg=UI["panel"])  # 예전 코드 호환 (표시 안 함)
        self.tank_head = head("IN TANK")
        self.tank_txt = tk.Text(side, height=1, font=(NUM_FONT, 12), relief="flat", bg=UI["panel"], fg=UI["ink"],
                                wrap="none", padx=14, pady=0, cursor="arrow", highlightthickness=0)
        self.tank_txt.pack(fill="x")
        self.tank_txt.tag_configure("big", font=(NUM_FONT, 20, "bold"))
        self.tank_txt.tag_configure("sub", foreground=UI["muted"], font=(NUM_FONT, 10))
        self.tank_txt.tag_configure("warn", foreground="#b7791f")
        self.tank_txt.tag_configure("over", foreground=UI["red"])
        head("ALERTS")
        self.alert_txt = tk.Text(side, height=7, font=(NUM_FONT, 11), relief="flat", bg=UI["panel"], fg=UI["ink"],
                                 wrap="word", padx=14, pady=0, cursor="arrow", highlightthickness=0)
        self.alert_txt.pack(fill="x")
        self.alert_txt.tag_configure("who", foreground=UI["red"], font=(NUM_FONT, 11, "bold"))
        self.alert_txt.tag_configure("ok", foreground=UI["green"])
        head("ZONES")
        self.zone_txt = tk.Text(side, height=7, font=(NUM_FONT, 10), relief="flat", bg=UI["panel"], fg=UI["ink"],
                                wrap="word", padx=14, pady=0, cursor="arrow", highlightthickness=0)
        self.zone_txt.pack(fill="x")
        self.zone_txt.tag_configure("name", font=(NUM_FONT, 11, "bold"))
        self.zone_txt.tag_configure("sub", foreground=UI["muted"])
        self.zone_txt.tag_configure("bad", foreground=UI["red"])
        head("EVENTS")
        self.log = tk.Text(side, font=(MONO_FONT, 8), relief="flat", bg="#f7f8fa", fg=UI["ink"], wrap="word", height=8,
                           padx=8, pady=6, highlightthickness=1, highlightbackground=UI["line"])
        self.log.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log.tag_configure("alarm", foreground=UI["red"], font=(MONO_FONT, 8, "bold"))
        self.log.tag_configure("t", foreground=UI["muted"])
        self.log.tag_configure("lv_alarm", foreground="white", background=UI["red"], font=(MONO_FONT, 8, "bold"))
        self.log.tag_configure("lv_info", foreground=UI["muted"])
        self.log.configure(state="disabled")

        self.status = tk.Label(self.root, anchor="w", bg=UI["panel"], fg=UI["muted"], font=(MONO_FONT, 9),
                               highlightbackground=UI["line"], highlightthickness=1)
        self.status.pack(fill="x", ipady=3, ipadx=10)
        self.setup_on = False

    def toggle_setup(self):
        self.setup_on = not self.setup_on
        if self.setup_on:
            self.setup_bar.pack(fill="x", after=self.root.winfo_children()[0])
            self.setup_hint.pack(fill="x", after=self.setup_bar, ipady=3, ipadx=12)
        else:
            self.setup_bar.pack_forget()
            self.setup_hint.pack_forget()
        self.setup_btn.config(text="CLOSE SETUP" if self.setup_on else "SETUP")

    def set_status(self, text):
        self.status.config(text=text)

    def on_limit(self):
        try:
            self.limit_min = max(1, int(self.limit_var.get()))
        except (tk.TclError, ValueError):
            pass

    def check_model(self):
        try:
            m = YOLO(str(MODEL_PATH))
            m.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
            dev = "GPU" if m.device.type == "cuda" else "CPU"
            ppe = "PPE model OK" if PPE_PATH.exists() else "No PPE model (ppe_best.pt) — PPE check off"
            try:
                self.wclip = WorkCLIP()
                wk = "work recognition (CLIP) on"
            except Exception as e:  # CLIP이 이 PC에 없으면 작업 추정만 끔
                wk = f"work recognition off ({type(e).__name__})"
            self.clip_done = True
            self.root.after(0, lambda: self.set_status(f"Ready ({dev}, {ppe}, {wk}) — click [Open video]"))
            self.root.after(0, lambda: self.ppe_lbl.config(text="Helmet · Goggles · Harness" if PPE_PATH.exists() else "No model"))
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Model error", msg))

    # ---------------- 영상 처리 (별도 스레드) ----------------
    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.start_video(path, paused=True)

    def start_video(self, path, paused):
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            messagebox.showerror("Video error", "Cannot open the video.\nIf the path contains non-English characters, move it to an English path.")
            return
        self.path, self.paused, self.ended = path, paused, False
        self.gen += 1
        with self.lock:
            self.zones = [z for z in self.zones if not getattr(z, "auto", False)]  # 영상마다 자동 전체 구역은 새로 만듦
            for z in self.zones:
                z.reset()
            self.reset_state()
        self.play_btn.config(text="PLAY" if paused else "PAUSE")
        self.add_log("info", f"Video started: {Path(path).name}", 0)
        threading.Thread(target=self.worker, args=(path, self.gen), daemon=True).start()

    def worker(self, path, gen):
        # 자세 추정 모델이 있으면 그것으로 추적 (사람 상자 + 코·눈·귀 위치를 한 번에) → 머리 기준 보호구 판정
        model = YOLO(str(POSE_PATH if POSE_PATH.exists() else MODEL_PATH))  # 영상마다 새로 만들어 추적 ID 초기화
        ppe = YOLO(str(PPE_PATH)) if PPE_PATH.exists() else None
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        blank = np.zeros((h or 640, w or 640, 3), np.uint8)
        model.predict(blank, imgsz=IMGSZ, verbose=False)  # 첫 실행 지연(수 초)을 시계 시작 전에 처리
        if ppe is not None:
            ppe.predict(np.zeros((320, 320, 3), np.uint8), imgsz=320, verbose=False)
        relinker = Relinker()
        # 작업 판단(CLIP) 준비 전에 재생하면 짧은 영상은 판단 없이 끝남(19초 계류 영상이 GENERAL WORK로 남던 원인) → 준비될 때까지 대기
        while not self.clip_done and gen == self.gen:
            self.root.after(0, lambda: self.set_status("Loading AI work recognition (CLIP)… playback starts automatically when ready"))
            time.sleep(0.5)
        idx, first, n_an, t_an, afps = 0, True, 0, time.perf_counter(), 0.0
        t0 = time.perf_counter()
        while gen == self.gen:
            if self.paused and not first:
                t0 = time.perf_counter() - idx / fps
                time.sleep(0.03)
                continue
            ok, frame = cap.read()
            if not ok:
                self.events.put(("end", "End of video", idx / fps))
                break
            idx += 1
            if not first and not self.all_frames_on and (time.perf_counter() - t0) - idx / fps > 0.15:
                continue  # 실시간 모드: 분석이 영상보다 느리면 프레임을 건너뛰어 실시간 유지 (모든 장면 모드는 건너뛰지 않음)
            if max(frame.shape[:2]) > MAX_SIDE:  # 휴대폰 4K 영상 → 1080p 수준으로 줄여 분석 (CCTV는 보통 그대로)
                s = MAX_SIDE / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            t = idx / fps
            self.cur_t = t
            if first and not self.zones:  # 구역을 안 그렸으면 화면 전체를 작업구역으로 → 바로 보호구 경보 (직접 그리면 사라짐)
                fh, fw = frame.shape[:2]
                z = Zone("FULL VIEW", "work", [(2, 2), (fw - 3, 2), (fw - 3, fh - 3), (2, fh - 3)], "GENERAL WORK")
                z.auto = True
                with self.lock:
                    self.zones.append(z)
                self.events.put(("info", "No zones drawn — whole view is monitored as GENERAL WORK (draw zones in SETUP to replace)", t))
            r = model.track(frame, persist=True, classes=[0], conf=DET_CONF, imgsz=IMGSZ,
                            tracker=str(TRACKER_YAML), verbose=False)[0]
            dets = []
            if r.boxes.id is not None:
                n = len(r.boxes)
                kps = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else [None] * n
                kcs = r.keypoints.conf.cpu().numpy() if r.keypoints is not None and r.keypoints.conf is not None else [None] * n
                dets = [(tid, box, (kp, kc, cf)) for tid, box, kp, kc, cf in
                        zip(r.boxes.id.int().tolist(), r.boxes.xyxy.int().tolist(), kps, kcs, r.boxes.conf.tolist())]
            reid = (lambda box, cands: self.reid(frame, box, cands)) if self.wclip is not None else None
            tracked = relinker.update(dets, t, reid)
            self.det_cf = {pid: cf for pid, _, (_, _, cf) in tracked}
            if ppe is not None:
                self.check_head(ppe, tracked, frame, t)
                self.check_hands(ppe, tracked, frame, t)
            persons = [(pid, tuple(b), ((b[0] + b[2]) // 2, b[3]), self.ppe_status(pid)) for pid, b, _ in tracked]
            if gen != self.gen:
                break
            self.update_zones(persons, frame, t)
            m = find_manhole(frame)
            if m is not None:
                if t - self.manhole_t > 0.5:
                    self.manhole_from = t  # 맨홀이 계속 보이기 시작한 시각
                self.manhole, self.manhole_t = m, t
                with self.lock:
                    for z in self.zones:
                        # 1초 이상 계속 보일 때만 (해치커버 영상: 15.5초에 한 번 스친 오탐으로 ENCLOSED 가 됐었음. C.mp4는 33초부터 계속 보임)
                        if (getattr(z, "auto", False) and getattr(z, "shown_work", None) != ENCLOSED
                                and t - self.manhole_from >= 1.0):
                            # 맨홀이 보이는 화면 = 밀폐구역 작업 (CLIP 추정보다 우선: C.mp4가 BOLT TIGHTENING 으로 남던 문제)
                            z.shown_work = ENCLOSED
                            self.events.put(("info", "Manhole detected — work shown as ENCLOSED SPACE", t))
            self.check_vanish(persons, frame, t)
            if self.wclip is not None:  # 프로그램 시작 직후엔 아직 불러오는 중일 수 있음
                self.classify_zones(self.wclip, frame, t)
                self.check_people(self.wclip, persons, frame, t)
            n_an += 1
            if time.perf_counter() - t_an >= 1:
                afps, n_an, t_an = n_an / (time.perf_counter() - t_an), 0, time.perf_counter()
            with self.lock:
                self.latest = {"frame": frame, "persons": persons, "t": t, "afps": afps, "fps": fps}
                self.new = True
            first = False
            ahead = idx / fps - (time.perf_counter() - t0)
            if ahead > 0 and not self.all_frames_on:
                time.sleep(ahead)
        cap.release()

    # ---------------- 보호구 판정 ----------------
    def vote(self, pid, item, v):
        hist = self.ppe_hist.setdefault(pid, {k: collections.deque(maxlen=PPE_HIST) for k in PPE_CLASSES})
        hist.setdefault(item, collections.deque(maxlen=PPE_HIST)).append(v)

    def check_head(self, ppe, tracked, frame, t):
        # 사람마다 0.5초에 1번: 머리만 잘라 확대 → 보호구 모델 (헬멧/맨머리/보안경). 헬멧을 못 찾으면 밝은 헬멧 색 비율
        H, W = frame.shape[:2]
        heads, pids, crowns, trusted, hairs = [], [], [], [], []
        for pid, box, (kp, kc, cf) in tracked:
            if t - self.head_t.get(pid, -1e9) < HEAD_SEC or box[3] - box[1] < 60:
                continue
            hb = head_box(box, kp, kc)
            if hb is None:
                continue
            x1, y1, x2, y2 = max(0, hb[0]), max(0, hb[1]), min(W, hb[2]), min(H, hb[3])
            head = frame[y1:y2, x1:x2]
            if head.size == 0 or x2 - x1 < 8:
                continue
            heads.append(head)
            pids.append(pid)
            face = kc is not None and sum(kc[i] > 0.5 for i in (0, 1, 2)) >= 2
            trusted.append(face or cf >= 0.5)  # 얼굴이 보이거나 사람 확신도 50%↑ 일 때만 '착용' 표 (계류 영상: 빨간 볼라드를 헬멧으로 봄)
            cb = crown_box(box, kp, kc) if hb[2] - hb[0] >= 45 else None  # 머리 45px 미만은 밝기 판정 안 함
            crown = frame[max(0, cb[1]):min(H, cb[3]), max(0, cb[0]):min(W, cb[2])] if cb else None
            crowns.append(crown if crown is not None and crown.size and cb[2] - cb[0] >= 6 else None)
            hairs.append(hair_dark(frame, hb) if hb[2] - hb[0] >= HAIR_MIN_PX else None)
            self.head_t[pid] = t
        if not heads:
            return
        big = [cv2.resize(c, (320, max(1, int(320 * c.shape[0] / c.shape[1]))), interpolation=cv2.INTER_CUBIC) for c in heads]
        for pid, head, crown, ok, hair, r in zip(pids, heads, crowns, trusted, hairs, ppe.predict(big, imgsz=320, conf=0.3, verbose=False)):
            found = {ppe.names[int(k)]: float(s) for k, s in zip(r.boxes.cls, r.boxes.conf)}
            if not ok:
                found.pop("helmet", None)  # 믿을 수 없는 탐지에서는 '착용' 근거로 쓰지 않음
                crown = None
            if ok and "helmet" not in found and hair is not None and hair[0] >= HAIR_DARK_N and hair[1] < HELMET_BRIGHT_Y:
                # 검은 머리(또는 짙은 모자·후드)만 보이고 헬멧 색 없음 → 미착용 (계류 영상: 맨머리 선원이 계속 '?'로 남던 문제)
                self.vote(pid, "helmet", "N")
            elif "helmet" in found or "no_helmet" in found:
                self.vote(pid, "helmet", "Y" if found.get("helmet", 0) >= found.get("no_helmet", 0) else "N")
            elif crown is not None:  # 얼굴이 보일 때만 정수리 밝기로 판정 (안 보이면 판정 보류)
                b = helmet_brightness(crown)
                if b >= HELMET_BRIGHT_Y:
                    self.vote(pid, "helmet", "Y")
                elif b <= HELMET_BRIGHT_N and helmet_brightness(head) < HELMET_BRIGHT_Y:
                    # 미착용은 머리 주변 전체에도 흰 헬멧 색이 없을 때만 (76초: 정수리 상자가 얼굴에 잡혀 흰 헬멧 착용자에게 오경보)
                    self.vote(pid, "helmet", "N")
            if "goggles" in found:
                self.vote(pid, "goggles", "Y")
                self.goggle_seen[pid] = t
            elif "no_goggle" in found:
                self.vote(pid, "goggles", "N")
            else:
                self.ppe_hist.setdefault(pid, {k: collections.deque(maxlen=PPE_HIST) for k in PPE_CLASSES})

    def check_hands(self, ppe, tracked, frame, t):
        # 장갑 필수 작업 구역 안 사람만, 1초에 1번: 사람 조각을 확대해 보호구 모델 → 손목(자세 추정) 근처의 장갑
        with self.lock:
            need = set().union(*[z.inside for z in self.zones if z.shown() in GLOVE_WORKS]) if self.zones else set()
        if not need:
            return
        H, W = frame.shape[:2]
        crops, metas = [], []
        for pid, (x1, y1, x2, y2), (kp, kc, cf) in tracked:
            if pid not in need or t - self.glove_t.get(pid, -1e9) < 1.0 or y2 - y1 < 60 or kc is None:
                continue
            wrists = [kp[i] for i in (9, 10) if kc[i] > 0.3]
            if not wrists:
                continue
            pw, ph = int((x2 - x1) * 0.15), int((y2 - y1) * 0.05)
            cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
            crop = frame[cy1:min(H, y2 + ph), cx1:min(W, x2 + pw)]
            if crop.size == 0:
                continue
            k = 640 / max(crop.shape[:2])
            crops.append(cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC))
            metas.append((pid, [((wx - cx1) * k, (wy - cy1) * k) for wx, wy in wrists], (y2 - y1) * k))
            self.glove_t[pid] = t
        if not crops:
            return
        for (pid, wr, hk), r in zip(metas, ppe.predict(crops, imgsz=640, conf=0.3, verbose=False)):
            for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
                if ppe.names[int(c)] != "gloves":
                    continue
                gx, gy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if any(((gx - wx) ** 2 + (gy - wy) ** 2) ** 0.5 <= 0.18 * hk for wx, wy in wr):  # 손목 근처
                    self.vote(pid, "gloves", "Y")
                    self.glove_seen[pid] = t
                    break

    def check_people(self, wclip, persons, frame, t):
        # 사람마다 1초에 1번: 겉모습 특징(같은 사람 다시 알아보기용) 갱신 + ALOFT 구역이면 하네스 판정
        with self.lock:
            aloft = set().union(*[z.inside for z in self.zones if z.work == "ALOFT"]) if self.zones else set()
        H, W = frame.shape[:2]
        boxes = {p[0]: p[1] for p in persons if p[1][3] - p[1][1] >= 60}

        def ov(a, b):  # a 상자 중 b와 겹친 비율
            area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
            return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])) / area

        def pad(b):  # 판정 조각과 같은 여백 (좌우 10%, 위아래 5%)
            pw, ph = (b[2] - b[0]) * 0.1, (b[3] - b[1]) * 0.05
            return (b[0] - pw, b[1] - ph, b[2] + pw, b[3] + ph)

        def near_of(a, b):  # b의 몸이 a의 판정 조각에 5% 넘게 들어오거나 서로 10% 넘게 겹침
            return ov(b, pad(a)) > 0.05 or ov(a, b) > 0.1 or ov(b, a) > 0.1

        due = [pid for pid in boxes if t - self.wear_t.get(pid, -1e9) >= HARNESS_SEC]
        if not due:
            return
        # 판정할 사람 + 그와 겹친 이웃(BA 점수 비교용) — 겹쳐도 건너뛰지 않음 (맨홀 앞 진입자는 보조자와 늘 겹침: C.mp4 진입자 F 210장 중 117장 누락)
        group = set(due) | {q for p in due for q in boxes if q != p and (near_of(boxes[p], boxes[q]) or near_of(boxes[q], boxes[p]))}
        pids = []
        crops = []
        for pid in group:
            x1, y1, x2, y2 = boxes[pid]
            pw, ph = int((x2 - x1) * 0.1), int((y2 - y1) * 0.05)
            c = frame[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)]
            if c.size:
                crops.append(c)
                pids.append(pid)
        if not crops:
            return
        embs = wclip.embed_crops(crops)
        hs = (embs @ wclip.pairs["harness"].T).cpu().numpy()
        bas = (embs @ wclip.pairs["ba"].T).cpu().numpy()
        embs = embs.cpu().numpy()
        ba_s = {pid: float(bp - bn) for pid, (bp, bn) in zip(pids, bas)}
        for pid, e, (sp, sn) in zip(pids, embs, hs):
            if pid not in due:
                continue
            over = max([ov(boxes[pid], boxes[q]) for q in boxes if q != pid] or [0])
            # 겹친 사람끼리는 BA 점수가 가장 높은 사람만 BA 표 (공기통이 보조자 상자에 걸쳐 보조자가 BA로 오판되던 문제 — C.mp4 노란 헬멧 A)
            rivals = [ba_s[q] for q in ba_s if q != pid and (near_of(boxes[pid], boxes[q]) or near_of(boxes[q], boxes[pid]))]
            top = ba_s[pid] > max(rivals or [-1])
            # 흐릿·가려진 탐지는 'BA 아님' 표 (표를 안 주면 예전 BA 표가 계속 남음 — 계류 영상 비트 뒤 사람 E)
            # 맨홀 근처(진입 후보)는 몸이 가려져 확신도가 낮음 → 기준 완화
            near = t - self.near_hatch.get(pid, -1e9) <= 2.0
            sure = self.det_cf.get(pid, 1.0) >= (BA_MIN_CONF_HATCH if near else BA_MIN_CONF)
            self.ba_hist.setdefault(pid, collections.deque(maxlen=PPE_HIST)).append(sure and top and ba_s[pid] > BA_TH)
            self.wear_t[pid] = t
            if over > 0.3:
                continue  # 겉모습 특징·하네스는 겹친 사람이 섞이므로 갱신 안 함 (예전 규칙 유지)
            old = self.app_emb.get(pid)
            m = e if old is None else 0.8 * old + 0.2 * e
            self.app_emb[pid] = m / np.linalg.norm(m)
            if pid in aloft:
                self.vote(pid, "harness", "Y" if sp - sn > HARNESS_TH else "N")

    def reid(self, frame, box, cands):
        """새로 잡힌 사람이 지금 안 보이는 사람 중 누구와 옷차림이 같은지 (같으면 그 이름을 이어 씀)"""
        cands = [k for k in cands if k in self.app_emb]
        if not cands:
            return None
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        c = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if c.size == 0:
            return None
        e = self.wclip.embed_crops([c]).cpu().numpy()[0]
        sims = sorted(((float(e @ self.app_emb[k]), k) for k in cands), reverse=True)
        sim, best = sims[0]
        margin = sim - sims[1][0] if len(sims) > 1 else 1.0
        if sim >= REID_SIM and margin >= REID_MARGIN:  # 두 번째 후보보다 확실히 비슷할 때만 (같은 작업복이면 이어 붙이지 않음)
            # 이름만 이어 쓰고 보호구는 처음부터 다시 확인 (잘못 이어져도 다른 사람의 '착용' 기록이 따라오지 않게)
            self.ppe_hist.pop(best, None)
            self.goggle_seen.pop(best, None)
            self.ba_hist.pop(best, None)       # BA도 다시 확인 (다른 사람의 BA가 따라오지 않게)
            self.ba_sure.discard(best)
            self.first_t[best] = self.cur_t
            self.events.put(("info", f"{pname(best)} recognized again (same clothing, similarity {sim:.2f}) — PPE re-checking", self.cur_t))
            return best
        return None

    def ppe_status(self, pid):
        out = {}
        for item, dq in self.ppe_hist.get(pid, {}).items():
            y, n = dq.count("Y"), dq.count("N")
            # 확실하지 않으면 초록 금지: 착용은 3표 이상이고 미착용 표의 2배 이상일 때만 / 미착용은 2표 이상이고 착용 표 이상
            out[item] = "Y" if y >= 3 and y >= 2 * n else "N" if n >= 2 and n >= y else None
        since = self.glove_since.get(pid)
        if since is not None and out.get("gloves") != "Y":  # 장갑 필수 작업: 들어온 뒤(또는 마지막 확인 뒤) 10초간 확인 안 되면 미착용
            out["gloves"] = "N" if self.cur_t - max(since, self.glove_seen.get(pid, -1e9)) >= GLOVE_MISSING_SEC else None
        ba = self.ba_hist.get(pid)
        if ba and sum(list(ba)[-BA_WIN:]) >= 2:
            # BA 착용: 최근 4번(약 4초) 중 기준 넘는 장면 2번 이상. 예전 9번은 흔들리는 화면에서 이름이 바뀐 뒤에도 BA가 남아
            # C.mp4에서 3명이 동시에 BA로 표시됨 → 4번으로 줄이니 동시 3명 0초, BA 표시 시간은 37→31초로 거의 유지
            out["goggles"] = "M"  # 'M' = 보안경은 확인 안 됐지만 BA 마스크로 대체 (초록으로 칠하지 않음 — 확인된 것처럼 보이면 안 됨)
            out["ba"] = "Y"
            return out
        if "goggles" in out and out["goggles"] != "Y" and self.ppe_on and pid in self.first_t:
            # 보안경은 작아서 '안 보임'이 흔함 → 화면에 나온 뒤(또는 마지막으로 보인 뒤) 10초간 한 번도 안 보이면 미착용
            since = max(self.first_t[pid], self.goggle_seen.get(pid, -1e9))
            if self.cur_t - since >= GOGGLE_MISSING_SEC:
                out["goggles"] = "N"
        return out

    # ---------------- 구역·작업 판단 ----------------
    def update_zones(self, persons, frame, t):
        feet = {p[0]: p[2] for p in persons}
        boxes = {p[0]: p[1] for p in persons}
        ppe = {p[0]: p[3] for p in persons}
        for pid, f in feet.items():
            self.first_seen.setdefault(pid, f)
            self.first_t.setdefault(pid, t)
        limit = self.limit_min * 60
        ev = self.events.put
        with self.lock:
            for z in self.zones:
                now_in = {pid for pid, f in feet.items() if z.contains(f)}
                from_hatch = lambda pid: z.work == ENCLOSED and z.contains(self.first_seen.get(pid))
                keep_time = z.kind == "work" and z.work != ENCLOSED  # 작업 시간 이어 세기 (밀폐구역은 탱크 타이머가 따로 있음)
                for pid in now_in:
                    z.last_feet[pid] = feet[pid]
                for k in [k for k, v in z.away.items() if t - v[1] > WORK_GRACE_SEC]:
                    z.away.pop(k)
                # 진입
                for pid in now_in:
                    z.miss.pop(pid, None)
                    if pid in z.inside:
                        continue
                    z.hits[pid] = z.hits.get(pid, 0) + 1
                    if z.hits[pid] >= ENTER_N:
                        z.inside.add(pid)
                        z.hits.pop(pid)
                        note = ""
                        if keep_time and pid in z.away:  # 잠깐 가려졌다 돌아옴 → 원래 시작 시각 그대로
                            z.enter_t[pid] = z.away.pop(pid)[0]
                            note = " — resumed"
                        elif keep_time and z.away:  # 새 번호로 다시 잡힘 → 근처에서 사라진 사람의 작업 시간을 넘겨받음
                            fx, fy = feet[pid]
                            k, v = min(z.away.items(), key=lambda kv: (kv[1][2][0] - fx) ** 2 + (kv[1][2][1] - fy) ** 2)
                            z.enter_t[pid] = v[0]
                            z.away.pop(k)
                            for item in PPE_KO:  # 같은 사람이므로 이미 낸 보호구 경보를 다시 내지 않음
                                if (z.name, k, item) in self.ppe_alarmed:
                                    self.ppe_alarmed.add((z.name, pid, item))
                            note = f" — continues {pname(k)} (same person, re-identified)"
                        else:
                            z.enter_t.setdefault(pid, t)
                        if z.kind == "danger":
                            ev(("alarm", f"DANGER: {pname(pid)} entered restricted zone {z.name}", t))
                        elif not from_hatch(pid):
                            ev(("info", f"{pname(pid)} entered {z.label()} (now {len(z.inside)}){note}", t))
                for pid in [p for p in z.hits if p not in now_in]:
                    z.hits.pop(pid)
                # 이탈 / 해치로 사라짐
                for pid in [p for p in z.inside if p not in now_in]:
                    z.miss[pid] = z.miss.get(pid, 0) + 1
                    if z.miss[pid] < EXIT_N:
                        continue
                    z.inside.discard(pid)
                    z.miss.pop(pid)
                    t_in = z.enter_t.pop(pid, t)
                    if keep_time:  # 60초 동안 작업 시간을 기억해 둠
                        z.away[pid] = (t_in, t, z.last_feet.get(pid, (0, 0)))
                    if z.work == ENCLOSED and pid not in feet and any(e["pid"] == pid for e in z.tank):
                        continue  # 이미 탱크 안으로 집계된 사람이 해치에서 잠깐 보였다 다시 내려감
                    if z.work == ENCLOSED and pid not in feet and (not from_hatch(pid) or not z.tank):
                        # 해치 구역 안에서 화면에서 사라짐 = 탱크로 내려감. 시간은 해치 구역에 들어선 순간부터
                        # (해치에서 처음 보인 사람은 탱크에서 올라온 사람으로 보지만, 탱크가 비어 있으면 새 진입으로 셈)
                        z.tank.append({"pid": pid, "name": None, "t_in": t_in, "warn": False, "over": False})
                        ev(("info", f"{pname(pid)} ENTERED TANK at {z.name} (inside: {len(z.tank)})", t))
                        if len(z.tank) == 1 or z.check_t is None or t - z.check_t > limit:
                            z.check_t = None
                            ev(("entry_check", (z, pname(pid)), t))
                        why = None
                        if z.gas_t is None:
                            why = "no passed gas test recorded"
                        elif t - z.gas_t > limit:
                            why = f"gas test older than {self.limit_min} min"
                        elif z.last_exit_t is not None and z.gas_t < z.last_exit_t:
                            why = "re-entry after a break without re-testing — IMO MSC.581(110) 8.3"
                        if why:
                            ev(("alarm", f"WARNING {z.name}: {pname(pid)} entered — {why}", t))
                    elif not from_hatch(pid):
                        ev(("info", f"{pname(pid)} left {z.label()} (now {len(z.inside)})", t))

                if z.work == ENCLOSED:
                    self.check_tank_exit(z, feet, boxes, now_in, t)
                    for e in z.tank:
                        el, who = t - e["t_in"], e["name"] or f"{pname(e['pid'])}"
                        if not e["warn"] and el >= limit * 0.8:
                            e["warn"] = True
                            ev(("info", f"{z.name}: {who} inside {mmss(el)} — must exit in {mmss(limit - el)}", t))
                        if not e["over"] and el >= limit:
                            e["over"] = True
                            ev(("alarm", f"DANGER {z.name}: {who} over {self.limit_min} min — EXIT NOW, re-test gas before re-entry "
                                         f"(company rule {self.limit_min} min, IMO 8.3)", t))

                if z.kind == "work":
                    if z.shown() in GLOVE_WORKS:
                        for pid in z.inside:
                            self.glove_since.setdefault(pid, t)
                    for pid in z.inside:
                        for item, v in ppe.get(pid, {}).items():
                            if item == "gloves" and z.shown() not in GLOVE_WORKS:
                                continue  # 장갑은 MOORING 등 필수 작업에서만
                            if item == "harness" and z.shown() != "ALOFT":
                                continue  # 하네스는 ALOFT 작업에서만 필수
                            key = (z.name, pid, item)
                            if v == "N" and key not in self.ppe_alarmed:
                                self.ppe_alarmed.add(key)
                                ev(("alarm", f"{PPE_WARN[item]}: {pname(pid)} — {z.label()}", t))
                    self.check_fire(z, frame, t)

    def check_vanish(self, persons, frame, t):
        # 밀폐구역 카메라에서, 화면 가장자리가 아닌 곳에서 2초 넘게 안 보이는 사람 → '진입 의심' 확인 창
        H, W = frame.shape[:2]
        hole = self.manhole if t - self.manhole_t <= MANHOLE_KEEP_SEC else None
        for pid, box, _, _ in persons:
            self.last_pos[pid] = (t, box)
            x1, y1, x2, y2 = box
            c = ((x1 + x2) / 2, (y1 + y2) / 2)
            if hole is not None and in_ellipse(c, hole, 1.0):
                self.on_hatch[pid] = t  # 몸 중심이 맨홀 안 (1.3배로 넓게 잡으면 옆에 선 사람까지 들어가 오경보)
            if hole is not None and (in_ellipse(c, hole, 1.3) or in_ellipse(((x1 + x2) / 2, y2), hole, 1.3)):
                self.near_hatch[pid] = t  # 맨홀 근처
        with self.lock:
            enclosed = any(z.work == ENCLOSED for z in self.zones)
            in_tank = {e["pid"] for z in self.zones for e in z.tank}
        for pid, (lt, (x1, y1, x2, y2)) in list(self.last_pos.items()):
            if t - lt < POSSIBLE_ENTRY_SEC:
                continue
            self.last_pos.pop(pid)
            hole_after = self.manhole_t >= lt + 0.3  # 사라진 뒤에도 맨홀이 보임 (그 사람 없이 비어 있음)
            near = lt - self.near_hatch.get(pid, -1e9) <= 1.5
            ba = self.ba_hist.get(pid)
            ba_like = bool(ba) and sum(ba) >= 2  # BA 멘 사람 = 진입 준비한 사람 (사용자 제안)
            strict = lt - self.on_hatch.get(pid, -1e9) <= 1.5
            dup = lt - self.last_auto_entry < 3.0 and not ba_like  # 직전 자동 진입 3초 안 + BA 없음 → 같은 진입의 중복(쪼개진 추적·돕던 사람)
            if pid not in in_tank and hole_after and (strict or near) and not dup:
                self.last_auto_entry = lt
                # 맨홀 안·근처에서 사라지고 그 뒤 맨홀이 비어 보임 → 무조건 자동 진입 (사용자 지시). 근거를 기록에 남김
                why = ("inside manhole" if strict else "near manhole") + (", BA" if ba_like else ", no BA confirmed")
                self.events.put(("auto_entry", (pid, lt, why), t))
                # (맨홀이 화면에서 벗어난 뒤 사라진 사람은 카메라가 움직여 빠진 것일 수 있어 제외 — C.mp4 42초 오경보)
                continue
            if dup and hole_after and (strict or near):
                continue  # 같은 진입의 중복 → 확인 창도 띄우지 않음
            edge = x1 < W * EDGE_MARGIN or x2 > W * (1 - EDGE_MARGIN) or y1 < H * EDGE_MARGIN  # 아래 끝은 제외 (맨홀은 발밑)
            if enclosed and not edge and pid not in in_tank and y2 - y1 >= 60:
                self.events.put(("possible_entry", (pid, lt), t))

    def classify_zones(self, wcls, frame, t):
        # 작업구역 화면을 잘라 작업 종류 추정 → 확률을 누적 평균해 한두 장면에 흔들리지 않게
        # (사람이 없어도 추정은 계속: 불꽃 판정에 씀. 작업 불일치 경보는 사람이 있을 때만)
        with self.lock:
            todo = [(z, cv2.boundingRect(z.poly)) for z in self.zones
                    if z.kind == "work" and t - z.cls_t >= WORK_CLS_SEC]
        crops, zs = [], []
        for z, (x, y, w, h) in todo:
            x, y = max(0, x), max(0, y)
            c = frame[y:y + h, x:x + w]
            if c.size:
                crops.append(c)
                zs.append(z)
        if not crops:
            return
        res = wcls.probs(crops)
        with self.lock:
            for z, p in zip(zs, res):
                z.cls_prob = p if z.cls_prob is None else 0.7 * z.cls_prob + 0.3 * p
                z.cls_t = t
                z.cls_n = getattr(z, "cls_n", 0) + 1
                est, conf = self.zone_estimate(z)
                if getattr(z, "auto", False) and getattr(z, "shown_work", None) != ENCLOSED and est != "NO WORK" and conf >= 0.6:
                    if getattr(z, "shown_work", None) != est:  # 자동 전체 구역: 카메라가 60% 이상 확신하는 작업으로 표시·규칙 적용
                        z.shown_work = est
                        self.events.put(("info", f"Camera recognizes the work as {est} ({conf:.0%})", t))
                # ALOFT 구역에 하네스 착용자가 있으면 그 자체가 고소작업의 근거 → 배경(바다·난간)만 보고 낸 추정은 무시
                harness_seen = z.work == "ALOFT" and any(self.ppe_status(p).get("harness") == "Y" for p in z.inside)
                if harness_seen:
                    continue
                if (z.inside and z.work in WORK_PROMPTS and est not in (z.work, "NO WORK") and conf >= WORK_MISMATCH
                        and not z.mismatch_alarmed):
                    z.mismatch_alarmed = True
                    self.events.put(("alarm", f"WARNING: WORK MISMATCH — {z.name}: permit {z.work}, camera sees {est} {conf:.0%}", t))
                elif est == z.work and conf >= 0.6:
                    z.mismatch_alarmed = False

    def zone_estimate(self, z):
        i = int(np.argmax(z.cls_prob))
        return self.wclip.names[i], float(z.cls_prob[i])

    def check_tank_exit(self, z, feet, boxes, now_in, t):
        # 해치에서 처음 나타난 사람(또는 탱크 안 사람과 같은 번호)이 해치 구역 밖에서 전신이 보이는 채로 3초 → 완전히 나옴
        if not z.tank:
            z.exit_cand.clear()
            return
        for pid, f in feet.items():
            same = next((e for e in z.tank if e["pid"] == pid), None)
            if same is None and not z.contains(self.first_seen.get(pid)):
                continue
            x1, y1, x2, y2 = boxes[pid]
            full_body = (y2 - y1) >= 1.6 * (x2 - x1)
            if pid in now_in or not full_body:
                z.exit_cand.pop(pid, None)
                continue
            start = z.exit_cand.setdefault(pid, t)
            if t - start < FULL_OUT_SEC:
                continue
            z.exit_cand.pop(pid)
            e = same or min(z.tank, key=lambda e: e["t_in"])  # 누구인지 모르면 가장 오래 있던 사람으로 처리
            z.tank.remove(e)
            z.last_exit_t = t  # 이후 재진입에는 새 가스 측정이 필요 (IMO 8.3)
            self.first_seen[pid] = f  # 다시 '해치에서 나온 사람'으로 세지 않도록
            who = e["name"] or f"{pname(e['pid'])}"
            note = "" if same else " (identity not confirmed — assumed longest inside, please verify)"
            self.events.put(("info", f"{who} EXITED TANK (fully out) at {z.name} — inside {mmss(t - e['t_in'])}, "
                                     f"still inside: {len(z.tank)}{note}", t))

    def check_fire(self, z, frame, t):
        # 용접 아크·불꽃: 구역 안에 '아주 밝고 흰' 점이 있고, 장면마다 깜빡이면 불꽃으로 봄 (햇빛 반사는 깜빡임이 적음)
        x, y, w, h = cv2.boundingRect(z.poly)
        x, y = max(0, x), max(0, y)
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        poly_mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(poly_mask, [z.poly - (x, y)], 1)
        m = (hsv[..., 2] >= 250) & (hsv[..., 1] <= 60) & (poly_mask > 0)
        ratio = m.mean()
        flick = np.logical_xor(m, z.prev_mask).mean() if z.prev_mask is not None and z.prev_mask.shape == m.shape else 0
        z.prev_mask = m
        hit = ratio > 0.0005 and flick > 0.0003
        if self.wclip is None or getattr(z, "cls_n", 0) < 2:
            hit = False  # CLIP 판정이 2번 이상 나오기 전에는 불꽃 판정 안 함 (시작 직후 계류 영상에서 바다 반짝임 오경보)
        else:
            # 흔들리는 화면 속 형광등·창문·바다 반짝임도 '밝게 깜빡임'으로 보일 수 있음 → CLIP도 용접 장면이라고 볼 때만
            hot_p = z.cls_prob[self.wclip.names.index("HOT WORK")]
            need = 0.5 if z.shown() == "MOORING" else FIRE_CLIP_MIN  # 계류 작업 중엔 바다 반짝임이 많아 기준을 높임
            hit = hit and hot_p >= need
        z.hot = 0.85 * z.hot + 0.15 * hit
        if not z.hot_on and z.hot > 0.5:
            z.hot_on = True
            if z.work == "HOT WORK":
                self.events.put(("info", f"{z.name}: welding sparks detected (hot work)", t))
            else:
                self.events.put(("alarm", f"DANGER: SPARKS DETECTED — {z.name}: permit is {z.work} (possible unauthorized hot work)", t))
        elif z.hot_on and z.hot < 0.15:
            z.hot_on = False
            self.events.put(("info", f"{z.name}: sparks stopped", t))

    # ---------------- 화면 갱신 (메인 스레드) ----------------
    def poll(self):
        while not self.events.empty():
            kind, text, t = self.events.get()
            if kind == "auto_entry":
                pid, t_gone, why = text
                self.add_entrant(self.enclosed_zone(), pname(pid), t_gone, pid=pid, how=f"auto: {why}")
                continue
            if kind == "possible_entry":
                self.ask_possible_entry(*text)
                continue
            if kind == "entry_check":
                self.open_entry_check(*text, t)
                continue
            if kind == "end":
                self.paused, self.ended = True, True
                self.play_btn.config(text="RESTART")
                self.add_log("info", "End of video", t)
            else:
                self.add_log(kind, text, t)
                if kind == "alarm":
                    self.raise_alarm(text)
        with self.lock:
            new, self.new = self.new, False
        if new:
            self.render()
            self.update_panel()
        self.root.after(30, self.poll)

    def render(self):
        with self.lock:
            latest = self.latest
            zones = [(i, z.kind, z.shown(), z.name, z.color(), z.poly.copy(), z.kind == "danger" and bool(z.inside),
                      [(e["name"] or pname(e["pid"]), e["t_in"]) for e in z.tank]) for i, z in enumerate(self.zones, 1)]
            danger_ids = set().union(*[z.inside for z in self.zones if z.kind == "danger"]) if self.zones else set()
            work_start = {}  # 작업 시작 시각: 작업구역에 들어선 시각 (작업구역 밖이면 처음 보인 시각)
            for z in self.zones:
                if z.kind == "work":
                    for p in z.inside:
                        if p in z.enter_t:
                            work_start[p] = min(work_start.get(p, 1e18), z.enter_t[p])
        if latest is None:
            return
        frame, persons, now = latest["frame"], latest["persons"], latest["t"]
        w, h = self.view.winfo_width(), self.view.winfo_height()
        if w < 20 or h < 20:
            return
        ih, iw = frame.shape[:2]
        s = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        self.view_tf = (s, (w - nw) / 2, (h - nh) / 2)
        limit = self.limit_min * 60

        # 구역: 얇은 선만 (위험구역은 점선, 사람이 들어가면 깜빡임)
        if any(occ for *_, occ, _ in zones):
            over = img.copy()
            for _, kind, _, _, col, poly, occ, _ in zones:
                if occ and self.flash:
                    cv2.fillPoly(over, [(poly * s).astype(np.int32)], BGR["bad"])
            img = cv2.addWeighted(over, 0.18, img, 0.82, 0)
        if self.manhole is not None and now - self.manhole_t <= MANHOLE_KEEP_SEC:
            (mx, my), (ma, mb), mang = self.manhole
            cv2.ellipse(img, ((mx * s, my * s), (ma * s, mb * s), mang), (0, 140, 255), 2, cv2.LINE_AA)
            hole_xy = (int((mx - ma / 2) * s), int((my - mb / 2) * s))
        else:
            hole_xy = None
        for _, kind, _, _, col, poly, occ, _ in zones:
            pts = (poly * s).astype(np.int32)
            if kind == "danger":
                dashed_poly(img, pts, BGR["bad"], 2)
            else:
                cv2.polylines(img, [pts], True, col, 1, cv2.LINE_AA)

        # 사람: 모서리 괄호 (추적 중 = 노랑, 보호구 모두 확인 = 초록, 미착용·위험구역 = 빨강)
        tags = []
        for pid, (x1, y1, x2, y2), (fx, fy), ppe in persons:
            bad = pid in danger_ids or "N" in ppe.values()
            ok = ppe and all(v in ("Y", "M") for v in ppe.values()) and "Y" in ppe.values()
            col = BGR["bad"] if bad else BGR["ok"] if ok else BGR["track"]
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            corners(img, p1, p2, col)
            started = work_start.get(pid, self.first_t.get(pid, now))
            tags.append((p1, pname(pid).split()[-1], mmss(now - started), ppe, col))

        if self.draw_kind and self.draw_pts:
            pts = (np.array(self.draw_pts) * s).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 200, 255), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(map(int, p)), 4, (0, 200, 255), -1, cv2.LINE_AA)

        band = img.copy()  # 위·아래 얇은 정보 띠
        cv2.rectangle(band, (0, 0), (nw, 24), (20, 24, 30), -1)
        cv2.rectangle(band, (0, nh - 22), (nw, nh), (20, 24, 30), -1)
        img = cv2.addWeighted(band, 0.5, img, 0.5, 0)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        white, navy, red = (255, 255, 255), (31, 58, 95), (198, 40, 40)
        txt = lambda xy, t, fill=white, bold=False: d.text(xy, t, font=font_for(t, bold), fill=fill)
        clock = time.strftime("%Y-%m-%d  %H:%M:%S")
        cw = d.textlength(clock, font=F_EN)
        txt((10, 4), "CAM 01", bold=True)
        txt((nw - cw - 10, 4), clock)
        d.ellipse((nw - cw - 60, 8, nw - cw - 51, 17), fill=(230, 40, 40) if int(time.time() * 2) % 2 else (110, 30, 30))
        txt((nw - cw - 46, 4), "REC", bold=True)
        txt((10, nh - 19), f"T+{mmss(now)}", fill=(200, 205, 214))

        placed = []
        if hole_xy is not None:  # 찾은 맨홀 이름표
            hx, hy = max(0, hole_xy[0]), max(26, hole_xy[1])
            d.rectangle((hx, hy, hx + d.textlength("MANHOLE", font=F_EN_B) + 12, hy + 18), fill=(255, 140, 0))
            txt((hx + 6, hy + 1), "MANHOLE", bold=True)
        for i, kind, work, name, col, poly, _, tank in zones:  # 구역: 작은 이름표 + 탱크 안 인원
            x, y = (poly * s).astype(int).min(axis=0)
            x, y = int(x), max(26, int(y))
            label = f"Z{i:02d}  {'RESTRICTED' if kind == 'danger' else work}"
            wl = d.textlength(label, font=F_EN_B) + 12
            d.rectangle((x, y, x + wl, y + 18), fill=red if kind == "danger" else col[::-1])
            txt((x + 6, y + 1), label, bold=True)
            for j, (who, t_in) in enumerate(tank):
                el = now - t_in
                row = f"IN TANK  {who}  {mmss(el)}"
                wr = d.textlength(row, font=font_for(row)) + 12
                fr = red if el >= limit else (183, 121, 31) if el >= limit * 0.8 else navy
                d.rectangle((x, y + 19 + 19 * j, x + wr, y + 37 + 19 * j), fill=fr)
                txt((x + 6, y + 20 + 19 * j), row)
            placed.append((x, y, x + wl, y + 19 + 19 * len(tank)))

        for p1, letter, worktime, ppe, col in tags:  # 사람: 'A  03:12' + 보호구 점 (경고 문구는 오른쪽 ALERTS 에만)
            code = {"helmet": "H", "goggles": "G", "harness": "HR", "ba": "BA", "gloves": "GL"}
            dots = [(code[k], {"Y": (46, 160, 90), "N": (220, 50, 50), "M": (40, 110, 200)}.get(v, (120, 126, 136)))
                    for k, v in ppe.items()]
            wid = (d.textlength(letter, font=F_EN_B) + d.textlength(worktime, font=F_EN) + 26
                   + sum(d.textlength(c, font=F_EN_B) + 10 for c, _ in dots))
            hgt = 18
            x = min(max(0, p1[0]), nw - wid)
            y = p1[1] - hgt - 3 if p1[1] - hgt - 3 > 26 else p1[1] + 3
            hit = lambda yy: next((r for r in placed if x < r[2] and x + wid > r[0] and yy < r[3] and yy + hgt > r[1]), None)
            for _ in range(len(placed) + 1):
                r = hit(y)
                if r is None:
                    break
                y = r[1] - hgt - 2 if r[1] - hgt - 2 > 26 else max(y, r[3] + 2)
            placed.append((x, y, x + wid, y + hgt))
            d.rectangle((x, y, x + wid, y + hgt), fill=(24, 30, 42))
            d.rectangle((x, y, x + 3, y + hgt), fill=col[::-1])
            txt((x + 8, y + 1), letter, bold=True)
            tx = x + 8 + d.textlength(letter, font=F_EN_B) + 8
            txt((tx, y + 1), worktime, fill=(205, 210, 220))
            dx = tx + d.textlength(worktime, font=F_EN) + 8
            for c, fill in dots:  # 글자 칩: 초록 = 확인됨, 빨강 = 미착용, 파랑 = BA 마스크로 대체, 회색 = 아직 확인 못 함
                cw2 = d.textlength(c, font=F_EN_B) + 6
                d.rectangle((dx, y + 3, dx + cw2, y + hgt - 3), fill=fill)
                d.text((dx + 3, y + 2), c, font=F_EN_B, fill=(255, 255, 255))
                dx += cw2 + 4
        self.tk_img = ImageTk.PhotoImage(pil)
        self.view.config(image=self.tk_img, text="")

        mode = (f"   DRAWING {'WORK' if self.draw_kind == 'work' else 'DANGER'} ZONE — "
                f"click = add point, right-click/Enter = finish, Esc = cancel") if self.draw_kind else ""
        self.set_status(f"{Path(self.path).name if self.path else ''}   T+{mmss(now)}   "
                        f"analysis {latest['afps']:.1f} fps   H helmet · G goggles · HR harness · BA breathing set   green = confirmed · red = missing · blue = covered by BA mask · grey = not confirmed{mode}")

    def update_panel(self):
        with self.lock:
            latest = self.latest
            if latest is None:
                return
            now, limit = latest["t"], self.limit_min * 60
            ppe = {p[0]: p[3] for p in latest["persons"]}
            alerts, zrows = [], []
            for i, z in enumerate(self.zones, 1):
                ids = sorted(z.inside)
                if z.kind == "danger":
                    alerts += [(pname(p), f"IN RESTRICTED ZONE {z.name}") for p in ids]
                    zrows.append((f"Z{i:02d}  {z.name}", f"RESTRICTED · {len(ids)} inside", bool(ids)))
                    continue
                for p in ids:
                    for k, v in ppe.get(p, {}).items():
                        if v == "N" and (k != "harness" or z.work == "ALOFT"):
                            alerts.append((pname(p), PPE_WARN[k].replace("WARNING: ", "")))
                if z.hot_on and z.work != "HOT WORK":
                    alerts.append((z.name, "SPARKS — POSSIBLE UNAUTHORIZED HOT WORK"))
                if z.mismatch_alarmed and z.cls_prob is not None:
                    alerts.append((z.name, f"WORK MISMATCH — camera sees {self.zone_estimate(z)[0]}"))
                sub = f"{z.shown()} · {len(ids)} people"
                if z.work == ENCLOSED:
                    sub = (f"{z.work} · {len(z.tank)} in tank · " + ("gas OK" if z.gas_t is not None else "NO GAS TEST")
                           + (" · procedure confirmed" if z.check_t is not None else " · PROCEDURE NOT CONFIRMED" if z.tank else ""))
                    if z.tank and z.check_t is None:
                        alerts.append((z.name, "ENTRY PROCEDURE NOT CONFIRMED"))
                    for e in z.tank:
                        el = now - e["t_in"]
                        if el >= limit:
                            alerts.append((e["name"] or pname(e["pid"]), f"IN TANK {mmss(el)} — EXIT NOW"))
                elif ids and z.cls_prob is not None:
                    est, conf = self.zone_estimate(z)
                    sub += f" · camera: {est} {conf:.0%}"
                zrows.append((f"Z{i:02d}  {z.name}", sub, False))
            names = [f"Z{i:02d}  {z.name}" for i, z in enumerate(self.zones, 1)]
        self.alerts = alerts
        self.total_lbl.config(text=f"{len(latest['persons'])} PEOPLE")
        with self.lock:
            tank = [(z.name, e["name"] or pname(e["pid"]), now - e["t_in"]) for z in self.zones for e in z.tank]
        tt = self.tank_txt
        tt.configure(state="normal")
        tt.delete("1.0", "end")
        if not tank:
            tt.insert("end", "Nobody inside", ("sub",))
        for zname, who, el in tank:  # 사람마다: 이름 · 경과 / 남은 시간 (80% 주황, 초과 빨강)
            left = limit - el
            tag = "over" if left <= 0 else "warn" if el >= limit * 0.8 else ""
            tt.insert("end", who + "\n", ("sub",))
            tt.insert("end", mmss(el), ("big", tag) if tag else ("big",))
            rest = f"OVER {mmss(-left)} — EXIT NOW" if left <= 0 else f"left {mmss(left)}"
            tt.insert("end", f"   {rest} / limit {self.limit_min} min\n", ("sub", tag) if tag else ("sub",))
        tt.configure(height=max(1, 2 * len(tank)), state="disabled")
        for widget, fill in ((self.alert_txt, alerts), (self.zone_txt, zrows)):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
        if alerts:
            for who, what in alerts:
                self.alert_txt.insert("end", "■ ", ("who",))
                self.alert_txt.insert("end", f"{who.upper()}  ", ("who",))
                self.alert_txt.insert("end", f"{what}\n")
        else:
            self.alert_txt.insert("end", "No alerts", ("ok",))
        for name, sub, bad in zrows:
            self.zone_txt.insert("end", name + "\n", ("name",))
            self.zone_txt.insert("end", sub + "\n\n", ("bad" if bad else "sub",))
        if not zrows:
            self.zone_txt.insert("end", "No zones — open SETUP to draw one", ("sub",))
        self.alert_txt.configure(state="disabled")
        self.zone_txt.configure(state="disabled")
        if list(self.zone_pick["values"]) != names:
            self.zone_pick["values"] = names
            if names and self.zone_pick.get() not in names:
                self.zone_pick.set(names[-1])

    def picked_zone(self):
        name = self.zone_pick.get()
        with self.lock:
            for i, z in enumerate(self.zones, 1):
                if f"Z{i:02d}  {z.name}" == name:
                    return z
        return None

    def add_log(self, kind, text, t):
        self.log.configure(state="normal")
        self.log.insert("1.0", text + "\n", ("alarm",) if kind == "alarm" else ())
        self.log.insert("1.0", " ALARM " if kind == "alarm" else " INFO  ", ("lv_alarm",) if kind == "alarm" else ("lv_info",))
        self.log.insert("1.0", f"T+{mmss(t)}  ", ("t",))
        self.log.configure(state="disabled")

    # ---------------- 경보 ----------------
    def raise_alarm(self, text):
        self.alarm_lbl.config(text=text)
        if not self.alarm_on:
            self.alarm_on = True
            self.alarm_bar.pack(fill="x", after=self.root.winfo_children()[0])
            self.blink()
        if winsound:
            threading.Thread(target=lambda: [winsound.Beep(f, 250) for f in (1000, 750, 1000)], daemon=True).start()

    def blink(self):
        if not self.alarm_on:
            self.flash = False
            return
        self.flash = not self.flash
        c = UI["red"] if self.flash else "#a51f1f"
        self.alarm_bar.config(bg=c)
        self.alarm_lbl.config(bg=c)
        self.render()
        self.root.after(500, self.blink)

    def ack_alarm(self):
        self.alarm_on = False
        self.alarm_bar.pack_forget()
        self.add_log("info", "Alarm acknowledged", self.latest["t"] if self.latest else 0)

    # ---------------- 밀폐구역: 수동 진입·퇴장 / 진입 의심 ----------------
    def enclosed_zone(self):
        """밀폐구역 구역 (없으면 수동 입력용 구역을 만듦 — 화면에는 이름표만)"""
        with self.lock:
            z = next((q for q in self.zones if q.work == ENCLOSED), None)
            if z is None:
                z = Zone("MANHOLE / TANK", "work", [(0, 0), (2, 0), (2, 2), (0, 2)], ENCLOSED)
                z.manual = True
                self.zones.append(z)
        return z

    def add_entrant(self, z, name, t_in, pid=-1, how="manual"):
        t_now = self.latest["t"] if self.latest else t_in
        with self.lock:
            z.tank.append({"pid": pid, "name": name, "t_in": t_in, "warn": False, "over": False})
            n = len(z.tank)
        self.add_log("info", f"{name} ENTERED TANK at {z.name} ({how}, inside: {n})", t_now)
        if n == 1 or z.check_t is None:
            z.check_t = None
            self.open_entry_check(z, name, t_in)
        self.update_panel()

    def manual_entry(self):
        if self.latest is None:
            messagebox.showinfo("Notice", "Open a video first.")
            return
        z = self.enclosed_zone()
        n = len(z.tank) + 1
        name = simpledialog.askstring("Manual entry", "Name of the person entering the tank", initialvalue=f"Entrant {n}",
                                      parent=self.root)
        if name:
            self.add_entrant(z, name.strip(), self.latest["t"], how="operator")

    def manual_exit(self):
        z = self.enclosed_zone()
        if not z.tank:
            messagebox.showinfo("Notice", "Nobody is recorded inside the tank.")
            return
        names = [e["name"] or pname(e["pid"]) for e in z.tank]
        pick = simpledialog.askinteger("Manual exit", "Who came out?\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1)),
                                       initialvalue=1, minvalue=1, maxvalue=len(names), parent=self.root)
        if pick:
            t = self.latest["t"] if self.latest else 0
            with self.lock:
                e = z.tank.pop(pick - 1)
                z.last_exit_t = t
            self.add_log("info", f"{names[pick - 1]} EXITED TANK at {z.name} (operator) — inside {mmss(t - e['t_in'])}", t)
            self.update_panel()

    def ask_possible_entry(self, pid, t_gone):
        win = tk.Toplevel(self.root)
        win.title("Possible tank entry")
        win.configure(bg=UI["panel"])
        win.attributes("-topmost", True)
        tk.Label(win, text="POSSIBLE ENCLOSED SPACE ENTRY", bg="#b7791f", fg="white", font=(NUM_FONT, 12, "bold"),
                 anchor="w", padx=14, pady=8).pack(fill="x")
        tk.Label(win, text=f"{pname(pid)} disappeared at T+{mmss(t_gone)} away from the screen edge.\n"
                           f"Did this person enter the tank?", bg=UI["panel"], fg=UI["ink"], font=(UI_FONT, 10),
                 justify="left").pack(anchor="w", padx=14, pady=10)
        f = tk.Frame(win, bg=UI["panel"])
        f.pack(fill="x", padx=14, pady=(0, 12))

        def yes():
            win.destroy()
            self.add_entrant(self.enclosed_zone(), pname(pid), t_gone, pid=pid, how="disappeared, operator confirmed")

        def no():
            win.destroy()
            self.add_log("info", f"{pname(pid)} disappearance at T+{mmss(t_gone)} — not an entry (operator)",
                         self.latest["t"] if self.latest else t_gone)
        tk.Button(f, text="CONFIRM ENTRY", command=yes, relief="flat", bg=UI["red"], fg="white",
                  font=(UI_FONT, 9, "bold"), padx=12, pady=4).pack(side="left")
        tk.Button(f, text="NOT AN ENTRY", command=no, relief="flat", bg="#f4f6f9", fg=UI["ink"],
                  font=(UI_FONT, 9, "bold"), padx=12, pady=4).pack(side="right")

    # ---------------- 밀폐구역: 진입 절차 확인표 ----------------
    def open_entry_check(self, z, who, t):
        if z.check_win is not None and z.check_win.winfo_exists():
            z.check_win.lift()
            return
        win = tk.Toplevel(self.root)
        z.check_win = win
        win.title("Enclosed space entry — procedure check")
        win.configure(bg=UI["panel"])
        win.attributes("-topmost", True)  # 영상은 계속 재생, 확인표는 맨 앞
        top = tk.Frame(win, bg=UI["red"])
        top.pack(fill="x")
        tk.Label(top, text="ENCLOSED SPACE ENTRY", bg=UI["red"], fg="white", font=(NUM_FONT, 14, "bold")).pack(
            anchor="w", padx=16, pady=(10, 0))
        timer = tk.Label(top, bg=UI["red"], fg="white", font=(NUM_FONT, 11))
        timer.pack(anchor="w", padx=16, pady=(0, 10))
        tk.Label(win, text="Confirm each item with the attendant / responsible officer (radio) — IMO MSC.581(110)",
                 bg=UI["panel"], fg=UI["muted"], font=(UI_FONT, 9)).pack(anchor="w", padx=16, pady=(10, 4))

        # 가스 측정: 마지막 기록을 그대로 보여 줌 (없거나 오래됐거나 퇴장 후 재측정 안 했으면 빨강)
        limit = self.limit_min * 60
        if z.gas_t is None:
            gas_txt, gas_ok = "Atmosphere tested — NO GAS TEST RECORDED", False
        else:
            v = z.gas_vals
            stale = t - z.gas_t > limit or (z.last_exit_t is not None and z.gas_t < z.last_exit_t)
            gas_txt = (f"Atmosphere tested at T+{mmss(z.gas_t)} — O₂ {v['O2']:g}%, CO₂ {v['CO2']:g} ppm, "
                       f"{v['LFL']:g}% LFL, H₂S {v['H2S']:g} ppm, CO {v['CO']:g} ppm" + ("  (OUT OF DATE)" if stale else ""))
            gas_ok = not stale
        box = tk.Frame(win, bg=UI["panel"])
        box.pack(fill="x", padx=16)
        vars_ = []
        btn_ok = None

        def refresh():  # 다 체크하기 전에는 회색 (눌러도 되는 것처럼 보이지 않게)
            done = all(v.get() for v in vars_)
            btn_ok.config(state="normal" if done else "disabled", bg=UI["green"] if done else "#c9ced6")

        for i, item in enumerate(ENTRY_CHECKS, 1):
            text = gas_txt if item == "GAS" else item
            v = tk.BooleanVar(value=False)
            vars_.append(v)
            tk.Checkbutton(box, text=f"{i:>2}. {text}", variable=v, command=lambda: refresh(), anchor="w",
                           justify="left", wraplength=640, bg=UI["panel"], activebackground=UI["panel"],
                           fg=UI["red"] if item == "GAS" and not gas_ok else UI["ink"],
                           font=(UI_FONT, 10, "bold" if item == "GAS" and not gas_ok else "normal")).pack(fill="x", pady=1)
        btns = tk.Frame(win, bg=UI["panel"])
        btns.pack(fill="x", padx=16, pady=14)

        def close():
            z.check_win = None
            win.destroy()

        def confirm():
            z.check_t = self.latest["t"] if self.latest else t
            self.add_log("info", f"{z.name}: entry procedure CONFIRMED by CCTV operator ({len(ENTRY_CHECKS)}/{len(ENTRY_CHECKS)} items)", z.check_t)
            close()
            self.update_panel()

        def not_ok():
            miss = [str(i) for i, v in enumerate(vars_, 1) if not v.get()]
            now = self.latest["t"] if self.latest else t
            msg = f"DANGER {z.name}: entry procedure NOT COMPLIANT — RECALL ENTRANTS (items {', '.join(miss) or '-'} not confirmed)"
            self.add_log("alarm", msg, now)
            self.raise_alarm(msg)
            close()

        btn_ok = tk.Button(btns, text="CONFIRM — ENTRY COMPLIANT", command=confirm, state="disabled", relief="flat",
                           bg="#c9ced6", fg="white", disabledforeground="#f4f6f9", font=(UI_FONT, 10, "bold"), padx=14, pady=6)
        btn_ok.pack(side="left")
        tk.Button(btns, text="NOT COMPLIANT — RECALL ENTRANTS", command=not_ok, relief="flat", bg=UI["red"], fg="white",
                  font=(UI_FONT, 10, "bold"), padx=14, pady=6).pack(side="right")
        win.protocol("WM_DELETE_WINDOW", close)  # 닫기만 하면 '미확인'으로 남아 ALERTS·경보 유지

        def tick():
            if not win.winfo_exists():
                return
            now = self.latest["t"] if self.latest else t
            timer.config(text=f"{z.name}  ·  {who} entered at T+{mmss(t)}  ·  in tank {mmss(now - t)}")
            win.after(500, tick)
        tick()

        def deadline():
            if z.check_t is None and (z.tank or win.winfo_exists()):
                msg = f"WARNING {z.name}: entry procedure not confirmed within {CHECK_DEADLINE_SEC} s"
                self.add_log("alarm", msg, self.latest["t"] if self.latest else t)
                self.raise_alarm(msg)
        self.root.after(CHECK_DEADLINE_SEC * 1000, deadline)

    # ---------------- 밀폐구역: 이름 · 가스 측정 ----------------
    def selected_enclosed(self):
        z = self.picked_zone()
        if z is None or z.work != ENCLOSED:
            with self.lock:
                z = next((q for q in self.zones if q.work == ENCLOSED), None)
        if z is None:
            messagebox.showinfo("Notice", "No ENCLOSED SPACE zone. Open SETUP and use [+ WORK ZONE] to mark the hatch entrance.")
        return z

    def name_tank(self):
        z = self.selected_enclosed()
        if z is None:
            return
        if not z.tank:
            messagebox.showinfo("Notice", f"{z.name}: nobody is in the tank.")
            return
        for e in list(z.tank):
            name = simpledialog.askstring("Tank entrant name", f"Name of {pname(e['pid'])} (entered {mmss(e['t_in'])})",
                                          initialvalue=e["name"] or "", parent=self.root)
            if name:
                e["name"] = name.strip()
                self.add_log("info", f"{z.name}: {pname(e['pid'])} → {e['name']}", self.latest["t"])
        self.update_panel()
        self.render()

    def record_gas(self):
        z = self.selected_enclosed()
        if z is None or self.latest is None:
            return
        vals = self.ask_gas(z.name)
        if vals is None:
            return
        t = self.latest["t"]
        bad = gas_failures(vals)
        if bad:  # 기준 미달: 기록은 남기되 진입 허용 안 함 (IMO 7.5: 추가 환기 후 재측정)
            self.add_log("alarm", f"DANGER {z.name}: GAS TEST FAILED — NO ENTRY ({', '.join(bad)}). Ventilate more and re-test", t)
            self.raise_alarm(f"DANGER {z.name}: GAS TEST FAILED — NO ENTRY")
        else:
            z.gas_t, z.gas_vals = t, vals
            self.add_log("info", f"{z.name}: gas test passed (IMO MSC.581(110) 7.4) — entry allowed for {self.limit_min} min", t)
        self.update_panel()

    def ask_gas(self, zone_name):
        win = tk.Toplevel(self.root)
        win.title("Gas test record")
        win.configure(bg="#fff")
        win.transient(self.root)
        win.grab_set()
        tk.Label(win, text=f"{zone_name} — readings before entry (steady values)", bg="#fff",
                 font=("맑은 고딕", 11, "bold")).grid(row=0, column=0, columnspan=3, padx=16, pady=(14, 8), sticky="w")
        crit = {"O2": "20.9", "CO2": "< 5,000", "LFL": "< 1",
                "H2S": f"< {OEL_PPM['H2S'] * 0.5:g}", "CO": f"< {OEL_PPM['CO'] * 0.5:g}"}
        ents = {}
        for i, (k, label, init) in enumerate(GAS_FIELDS, 1):
            tk.Label(win, text=label, bg="#fff", font=("맑은 고딕", 10)).grid(row=i, column=0, sticky="w", padx=16, pady=3)
            e = tk.Entry(win, width=10, font=("맑은 고딕", 11))
            e.insert(0, init)
            e.grid(row=i, column=1, padx=6)
            tk.Label(win, text=f"req. {crit[k]}", bg="#fff", fg="#5b6578").grid(row=i, column=2, sticky="w", padx=(0, 16))
            ents[k] = e
        tk.Label(win, text="IMO MSC.581(110) 7.4 · toxic gas limit = 50% of OEL", bg="#fff", fg="#8a93a3",
                 font=("맑은 고딕", 9)).grid(row=len(GAS_FIELDS) + 1, column=0, columnspan=3, padx=16, sticky="w")
        res = {"v": None}

        def ok():
            try:
                res["v"] = {k: float(e.get()) for k, e in ents.items()}
            except ValueError:
                messagebox.showwarning("Check input", "Please enter numbers.", parent=win)
                return
            win.destroy()
        tk.Button(win, text="Record", font=("맑은 고딕", 10), padx=16, command=ok).grid(
            row=len(GAS_FIELDS) + 2, column=0, columnspan=3, pady=12)
        self.root.wait_window(win)
        return res["v"]

    # ---------------- 재생 / 구역 그리기 ----------------
    def toggle_play(self):
        if not self.path or self.draw_kind:
            return
        if self.ended:
            self.start_video(self.path, paused=False)
            return
        self.paused = not self.paused
        self.play_btn.config(text="PLAY" if self.paused else "PAUSE")

    def start_draw(self, kind):
        if self.latest is None:
            messagebox.showinfo("Notice", "Open a video first.")
            return
        self.paused = True
        self.play_btn.config(text="PLAY")
        self.draw_kind, self.draw_pts = kind, []
        self.render()

    def on_click(self, e):
        if not self.draw_kind:
            return
        s, ox, oy = self.view_tf
        ih, iw = self.latest["frame"].shape[:2]
        x, y = (e.x - ox) / s, (e.y - oy) / s
        if 0 <= x < iw and 0 <= y < ih:
            self.draw_pts.append((x, y))
            self.render()

    def ask_work(self):
        win = tk.Toplevel(self.root)
        win.title("Work type")
        win.configure(bg="#fff")
        win.transient(self.root)
        win.grab_set()
        tk.Label(win, text="Work in this zone", bg="#fff", font=("맑은 고딕", 11, "bold")).pack(padx=20, pady=(14, 6))
        var, res = tk.StringVar(value=WORK_TYPES[0]), {"v": None}
        for w in WORK_TYPES:
            tip = "  (draw the hatch entrance)" if w == ENCLOSED else ""
            tk.Radiobutton(win, text=w + tip, variable=var, value=w, bg="#fff",
                           font=("맑은 고딕", 11)).pack(anchor="w", padx=24)
        tk.Button(win, text="OK", font=("맑은 고딕", 10), padx=16,
                  command=lambda: (res.update(v=var.get()), win.destroy())).pack(pady=12)
        self.root.wait_window(win)
        return res["v"]

    def finish_draw(self):
        if not self.draw_kind:
            return
        if len(self.draw_pts) < 3:
            messagebox.showinfo("Notice", "Click at least 3 points.")
            return
        work = None
        if self.draw_kind == "work":
            work = self.ask_work()
            if work is None:
                return
        n = sum(1 for z in self.zones if z.kind == self.draw_kind) + 1
        default = f"{work} zone {n}" if work else f"Danger zone {n}"
        name = simpledialog.askstring("Zone name", "Zone name (e.g. FWD mooring deck, No.3 COT hatch, snap-back zone)",
                                      initialvalue=default, parent=self.root) or default
        z = Zone(name, self.draw_kind, self.draw_pts, work)
        with self.lock:
            self.zones = [q for q in self.zones if not getattr(q, "auto", False)] + [z]  # 자동 전체 구역은 직접 그린 구역으로 교체
        self.add_log("info", f"Zone added: {z.label()}", self.latest["t"])
        self.cancel_draw()
        self.update_panel()

    def cancel_draw(self):
        self.draw_kind, self.draw_pts = None, []
        self.render()

    def delete_zone(self):
        z = self.picked_zone()
        if z is None:
            messagebox.showinfo("Notice", "Pick the zone to delete in ZONE.")
            return
        with self.lock:
            self.zones = [q for q in self.zones if q is not z]
        self.zone_pick.set("")
        self.update_panel()
        self.render()

    def save_zones(self):
        if not self.zones or self.latest is None:
            messagebox.showinfo("Notice", "No zones to save.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".json", initialfile="zones.json",
                                           filetypes=[("Zone settings", "*.json")])
        if out:
            ih, iw = self.latest["frame"].shape[:2]
            data = {"size": [iw, ih],
                    "zones": [{"name": z.name, "kind": z.kind, "work": z.work, "pts": z.pts} for z in self.zones]}
            Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            self.set_status(f"Zones saved: {out}")

    def load_zones(self):
        path = filedialog.askopenfilename(filetypes=[("Zone settings", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            zones = [Zone(z["name"], z["kind"], z["pts"], z.get("work")) for z in data["zones"]]
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read the zone file.\n{e}")
            return
        if self.latest is not None and list(self.latest["frame"].shape[1::-1]) != data.get("size"):
            messagebox.showwarning("Caution", "Video size differs from when zones were saved — zones may be misaligned.")
        with self.lock:
            self.zones = zones  # 불러온 구역으로 교체 (자동 전체 구역 포함 기존 구역 대체)
        self.update_panel()
        self.render()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Ship Work Camera Monitor — Work type · PPE · Enclosed space")
    root.geometry("1560x900")
    App(root)
    root.mainloop()
