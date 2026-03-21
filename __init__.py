"""
Cross-BiMagnitude Stitching Nodes for ComfyUI

고차통계(HOS)의 교차 바이스펙트럼(Cross-Bispectrum) 크기를 이용한
이미지 정합·스티칭 커스텀 노드 패키지입니다.

설치 방법:
    ComfyUI/custom_nodes/ 폴더 아래에 이 패키지 폴더를 통째로 복사하세요.
    (폴더 이름 예: cross_bimagnitude_stitch)

필요 패키지:
    numpy, scipy, torch  (ComfyUI 기본 환경에 포함)
"""

from .cross_bimagnitude_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
