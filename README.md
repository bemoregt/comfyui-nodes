# ComfyUI-LSD-DCT-Curve

이미지 속 개곡선(선/획/경로)을 LSD(Line Segment Detector)로 검출하고, 검출된 직선 조각들을 하나의 곡선으로 체이닝한 뒤 DCT로 스무딩/압축하는 ComfyUI 커스텀 노드입니다.

## 설치

1. 이 폴더 전체를 `ComfyUI/custom_nodes/` 아래에 복사합니다.

   ```
   ComfyUI/custom_nodes/ComfyUI-LSD-DCT-Curve/
   ```

2. ComfyUI가 사용하는 Python 환경에서 의존성을 설치합니다.

   ```
   pip install -r ComfyUI/custom_nodes/ComfyUI-LSD-DCT-Curve/requirements.txt
   ```

   `opencv-contrib-python`은 `cv2.ximgproc.createFastLineDetector` (FLD) 폴백에 필요합니다. 이미 `opencv-python`이 설치되어 있다면 충돌을 피하기 위해 먼저 제거 후 설치하세요.

   ```
   pip uninstall opencv-python
   pip install opencv-contrib-python
   ```

3. ComfyUI를 재시작하면 노드 목록의 `curve/detection` 카테고리에 **"LSD -> DCT Curve Smoother"** 노드가 나타납니다.

## 노드: LSD -> DCT Curve Smoother

**입력**
- `image` (IMAGE): 개곡선이 포함된 이미지

**위젯**
| 이름 | 설명 | 기본값 |
|---|---|---|
| `num_dct_coeffs` | 저주파부터 남길 DCT 계수 개수. 작을수록 더 많이 스무딩/압축됨 | 16 |
| `num_samples` | DCT 전에 곡선을 균일 리샘플링할 점 개수 | 256 |
| `min_segment_length` | 이보다 짧은 LSD 조각(잡음)은 무시 | 5.0 |
| `max_gap` | 체이닝 시 허용하는 최대 끝점 간격. `0`이면 무제한(이미지 전체를 하나의 곡선으로 간주) | 0.0 |
| `canvas_mode` | 결과 곡선을 그릴 배경: `overlay_on_original` / `blank_black` / `blank_white` | overlay_on_original |
| `line_thickness` | 그려지는 곡선 두께(px) | 2 |
| `line_color` | `"R,G,B"` 형식의 곡선 색상 | 255,0,0 |

**출력**
- `image` (IMAGE): 스무딩된 곡선이 그려진 이미지
- `curve_json` (STRING): 검출/체이닝/DCT 결과를 담은 JSON. 필드:
  - `engine`: 실제 사용된 검출 엔진 (`"LSD"` 또는 `"FLD"`)
  - `num_raw_segments`: LSD/FLD가 검출한 직선 조각 개수
  - `num_chained_points`: 체이닝 후 폴리라인 점 개수
  - `num_samples`, `num_dct_coeffs_kept`
  - `smoothed_curve`: 스무딩된 곡선의 `[x, y]` 좌표 배열
  - `dct_coeffs_x`, `dct_coeffs_y`: 남겨진 DCT 계수 (다른 노드에서 재구성/비교용으로 활용 가능)

## 동작 원리

1. **검출**: `cv2.createLineSegmentDetector`(LSD)를 우선 시도하고, 특허 이슈로 해당 API가 없는 OpenCV 빌드에서는 `cv2.ximgproc.createFastLineDetector`(FLD)로 자동 폴백합니다.
2. **체이닝**: LSD/FLD는 곡선을 여러 개의 짧은 직선 조각으로 반환하므로, 인접한 조각의 끝점을 그리디하게 최근접 연결해 하나의 순서 있는 폴리라인으로 합칩니다.
3. **리샘플링**: DCT는 균일 샘플링을 전제하므로, 체이닝된 폴리라인을 호 길이(arc length) 기준으로 `num_samples`개 점으로 다시 뽑습니다.
4. **DCT 스무딩/압축**: x(t), y(t) 시퀀스에 각각 DCT-II를 적용하고 저주파 `num_dct_coeffs`개만 남긴 뒤 역변환해 잡음이 제거된 곡선을 복원합니다.

## 알려진 제한

- 여러 개의 분리된 곡선이 한 이미지에 있는 경우, 현재 버전은 모든 조각을 하나의 곡선으로 체이닝합니다(설계상 가정). 분리된 곡선을 각각 다루려면 `max_gap`을 낮추거나 별도 배치 처리가 필요합니다.
- DCT 계수를 적게 남길수록 곡선 양 끝단에 약간의 처짐/리플이 생길 수 있습니다(경계 처리 특성).

## 실행 예시

실제 ComfyUI에서 `이미지 로드 → LSD -> DCT Curve Smoother → 이미지 미리보기`로 연결해 실행한 결과입니다. 검정 선이 원본, 빨간 선이 LSD로 검출·체이닝 후 DCT로 스무딩된 곡선입니다.

| | |
|---|---|
| ![Z자 곡선 예시](ScrShot%2015.png) `num_dct_coeffs=7` — 꺾인 Z자 모양도 적은 계수로 형태를 보존 | ![중첩 사각형 예시](ScrShot%2016.png) `num_dct_coeffs=39` — 중첩된 사각형 개곡선(G자 형태)도 안정적으로 추종 |
| ![여러 개의 분리된 곡선 예시](ScrShot%2017.png) `num_dct_coeffs=39` — 서로 떨어진 곡선 여러 개가 하나의 폴리라인으로 체이닝되는 알려진 제한을 보여주는 예시 | ![S자 곡선 예시](ScrShot%2018.png) `num_dct_coeffs=37` — 굴곡이 많은 S자 곡선도 노이즈 없이 매끄럽게 복원 |
