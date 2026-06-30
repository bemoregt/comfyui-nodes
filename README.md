# EVM Magnification — ComfyUI Custom Node

오일러 비디오 확대(Eulerian Video Magnification) 기법을 ComfyUI 커스텀 노드로 구현한 패키지입니다.  
육안으로 보이지 않는 미세한 움직임(미세 표정, 맥박, 호흡 등)을 증폭하여 영상으로 시각화합니다.

> **논문 출처**  
> Wu et al., *"Eulerian Video Magnification for Revealing Subtle Changes in the World"*  
> ACM SIGGRAPH 2012

---

## 오일러(Eulerian) 관점이란?

| 관점 | 방식 | 문제점 |
|---|---|---|
| **라그랑주** (Lagrangian) | 픽셀·특징점을 직접 **추적** | 미세 움직임은 픽셀 1~2개 수준이라 노이즈에 묻힘 |
| **오일러** (Eulerian) | 고정된 픽셀 위치에서 **밝기 변화를 시간축으로 관찰** | 추적 없이 주파수 필터링만으로 증폭 가능 |

EVM은 오일러 관점을 채택합니다.  
"픽셀이 어디로 이동했는가?"를 찾는 대신, "고정된 픽셀의 밝기가 시간에 따라 어떻게 진동하는가?"를 분석하고 그 진동을 증폭합니다.

---

## 알고리즘 파이프라인

```
입력 프레임 (T, H, W, C)
        │
        ▼
[1] 공간 분해 — 라플라시안 피라미드 (levels 단계)
        │  각 레벨: 특정 공간 주파수 대역의 디테일
        ▼
[2] 시간 필터링 — 각 피라미드 레벨에서 시간축으로 Butterworth 밴드패스 적용
        │  low_freq ~ high_freq 사이의 진동만 통과
        ▼
[3] 증폭 — 필터링된 신호 × alpha
        ▼
[4] 재합성 — 피라미드 역변환 후 원본 프레임에 덧셈
        │
        ▼
출력 프레임 (T, H, W, C)  ← 미세 모션이 증폭된 영상
```

핵심 수식 (1차 테일러 근사):

```
I(x, t) ≈ f(x) + δ(x,t)·∂f/∂x
증폭 후  ≈ f(x) + (1+α)·δ_filtered(x,t)·∂f/∂x
         ≈ f(x + (1+α)·δ_filtered)    ← 움직임이 (1+α)배 확대된 것과 동일
```

---

## 설치

### 1. 패키지 위치
```
ComfyUI/custom_nodes/evm_magnification/   (또는 심볼릭 링크)
```

본 저장소를 직접 `custom_nodes` 폴더 안에 두거나, 심볼릭 링크를 생성합니다.

```bash
ln -s /path/to/0630_evm_node \
      /path/to/ComfyUI/custom_nodes/evm_magnification
```

### 2. 의존 패키지
```bash
pip install opencv-python numpy scipy
```

또는:
```bash
pip install -r requirements.txt
```

---

## 노드 설명

ComfyUI 노드 검색창에서 **`EVM`** 으로 검색하면 `video/EVM` 카테고리 아래에 두 노드가 나타납니다.

---

### EVM Magnification (Video File)

비디오 파일 경로를 직접 입력받아 증폭된 MP4를 ComfyUI output 폴더에 저장합니다.

| 입력 | 타입 | 설명 |
|---|---|---|
| `video_path` | STRING | 입력 비디오의 절대 경로 |
| `alpha` | FLOAT | 증폭 배율 (기본 50) |
| `low_freq` | FLOAT | 밴드패스 하한 주파수 (Hz) |
| `high_freq` | FLOAT | 밴드패스 상한 주파수 (Hz) |
| `pyramid_levels` | INT | 라플라시안 피라미드 단계 수 |
| `filter_order` | INT | 버터워스 필터 차수 |
| `color_space` | rgb / yiq | 색상 공간 선택 |
| `output_filename` | STRING | 출력 파일명 (확장자 제외) |

| 출력 | 타입 | 설명 |
|---|---|---|
| `output_video_path` | STRING | 저장된 MP4 파일의 절대 경로 |

---

### EVM Magnification (Frames)

ComfyUI의 IMAGE 배치(프레임 텐서)를 입력받아 증폭된 IMAGE 배치를 반환합니다.  
VHS(Video Helper Suite) 등 다른 비디오 노드와 연결해 사용합니다.

```
VHS LoadVideo ──► EVM Magnification (Frames) ──► VHS SaveVideo
```

| 입력 | 타입 | 설명 |
|---|---|---|
| `images` | IMAGE | 프레임 배치 텐서 (B, H, W, C) |
| `fps` | FLOAT | 원본 영상의 프레임 레이트 |
| (나머지 파라미터) | — | Video File 노드와 동일 |

| 출력 | 타입 | 설명 |
|---|---|---|
| `images` | IMAGE | 증폭된 프레임 배치 텐서 |

---

## 파라미터 설정 가이드

### 용도별 추천 설정

| 용도 | `alpha` | `low_freq` | `high_freq` | `color_space` |
|---|---|---|---|---|
| 미세 표정 (Micro-expression) | 30 – 80 | 0.5 | 4.0 | rgb |
| 심박수 (맥박 색상 변화) | 100 – 200 | 0.8 | 1.2 | yiq |
| 호흡 움직임 | 20 – 50 | 0.1 | 0.4 | rgb |
| 손 떨림 | 20 – 60 | 1.0 | 10.0 | rgb |

### 파라미터별 설명

**`alpha` (증폭 배율)**  
크게 설정할수록 미세한 움직임이 강조됩니다. 너무 높으면 노이즈까지 증폭되어 영상이 깨집니다.  
처음에는 20~50으로 시작해 점차 높여가며 조절하세요.

**`low_freq` / `high_freq` (밴드패스 주파수 Hz)**  
관심 있는 움직임의 주파수 대역을 지정합니다.  
`fps / 2`(나이퀴스트 주파수)를 초과하지 않도록 설정하세요.

**`pyramid_levels` (피라미드 단계)**  
단계가 많을수록 더 넓은 스케일 범위의 움직임을 포착합니다.  
해상도가 낮을수록 단계 수를 줄여야 합니다 (최소 크기가 8×8 이상 유지되도록).

**`filter_order` (필터 차수)**  
차수가 높을수록 밴드패스가 급격하게 잘리지만, 짧은 클립에서는 링잉(ringing) 아티팩트가 생길 수 있습니다.  
짧은 영상(< 60프레임)은 1을 권장합니다.

**`color_space`**  
- `rgb` : 밝기·색상 모든 채널을 동일하게 증폭 → 모션 증폭에 적합  
- `yiq` : 휘도(Y)와 색차(I, Q)를 분리해 처리 → 색상 변화(맥박 등) 증폭에 적합

---

## 파일 구조

```
0630_evm_node/
├── __init__.py        # ComfyUI 노드 등록 (NODE_CLASS_MAPPINGS)
├── evm_processor.py   # EVM 핵심 알고리즘
│                        - 라플라시안 피라미드 빌드/재합성
│                        - Butterworth 밴드패스 필터 설계
│                        - 시간축 필터링 (filtfilt)
│                        - YIQ 색상 공간 변환
├── nodes.py           # ComfyUI 노드 클래스 정의
│                        - EVMMagnification  (Frames)
│                        - EVMVideoFile      (Video File)
├── requirements.txt   # 의존 패키지 목록
└── README.md
```

---

## 의존 패키지

| 패키지 | 용도 |
|---|---|
| `opencv-python` | 비디오 I/O, 피라미드 연산 (`pyrDown` / `pyrUp`) |
| `numpy` | 배열 연산 |
| `scipy` | Butterworth 필터 설계 및 `filtfilt` 적용 |
