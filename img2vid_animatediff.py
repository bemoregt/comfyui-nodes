#!/usr/bin/env python3
"""
ComfyUI API를 통해 AnimateDiff로 이미지 -> 비디오 클립을 생성하는 스크립트.

전제 조건 (ComfyUI에 아래 커스텀 노드가 설치되어 있어야 함):
  - ComfyUI-AnimateDiff-Evolved  (ADE_AnimateDiffLoaderWithContext 노드)
  - ComfyUI-VideoHelperSuite     (VHS_VideoCombine 노드)

IPAdapter는 사용하지 않는 순수 AnimateDiff 버전입니다. 입력 이미지를 latent로
인코딩해 프레임 수만큼 복제한 뒤 denoise<1.0으로 샘플링하여 원본과의 연속성을
유지합니다.

필요 패키지: pip install requests websocket-client

주의: 커스텀 노드는 버전에 따라 class_type 이름이나 입력 파라미터가 바뀔 수 있습니다.
ComfyUI 실행 후 큐가 실패하면(400/에러 로그) 아래 build_workflow() 안의
해당 노드 class_type/inputs를 본인 ComfyUI에 설치된 버전에 맞게 수정하세요.
(ComfyUI 메뉴의 "Save (API Format)"으로 실제 워크플로우를 export해서
 노드 이름을 비교해보면 가장 정확합니다.)
"""

import argparse
import json
import mimetypes
import os
import random
import time
import uuid

import requests
import websocket  # pip install websocket-client


def upload_image(server_address: str, image_path: str) -> str:
    """ComfyUI 서버에 입력 이미지를 업로드하고, 서버 내부 파일명을 반환한다."""
    url = f"http://{server_address}/upload/image"
    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, content_type)}
        data = {"overwrite": "true"}
        resp = requests.post(url, files=files, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["name"]


def build_workflow(cfg: dict, uploaded_image_name: str) -> dict:
    """순수 AnimateDiff 기반 image-to-video 워크플로우(API 형식)를 구성한다."""

    workflow = {
        # 1. 체크포인트 로드
        "checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": cfg["checkpoint"]},
        },
        # 2. 입력 이미지 로드 (업로드된 파일 참조)
        "load_image": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image_name},
        },
        # 3. AnimateDiff 모션 모듈을 모델에 적용
        "animatediff_loader": {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model": ["checkpoint", 0],
                "model_name": cfg["motion_module"],
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "motion_scale": 1.0,
                "apply_v2_models_properly": True,
                "context_options": ["context_options", 0],
            },
        },
        "context_options": {
            "class_type": "ADE_StandardStaticContextOptions",
            "inputs": {
                "context_length": min(cfg["frames"], 16),
                "context_stride": 1,
                "context_overlap": 4,
                "closed_loop": False,
                "fuse_method": "flat",
            },
        },
        # 4. 프롬프트 인코딩
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": cfg["prompt"], "clip": ["checkpoint", 1]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": cfg["negative_prompt"], "clip": ["checkpoint", 1]},
        },
        # 6. 입력 이미지를 latent로 인코딩 후, 프레임 수만큼 복제 (img2vid 시작점)
        "vae_encode": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["load_image", 0], "vae": ["checkpoint", 2]},
        },
        "repeat_latent": {
            "class_type": "RepeatLatentBatch",
            "inputs": {"samples": ["vae_encode", 0], "amount": cfg["frames"]},
        },
        # 7. 샘플링 (denoise < 1.0 으로 원본 이미지와의 연속성 유지)
        "ksampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["animatediff_loader", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["repeat_latent", 0],
                "seed": cfg["seed"],
                "steps": cfg["steps"],
                "cfg": cfg["cfg_scale"],
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": cfg["denoise"],
            },
        },
        # 8. latent -> 이미지 디코딩
        "vae_decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["ksampler", 0], "vae": ["checkpoint", 2]},
        },
        # 9. 프레임들을 비디오 파일로 저장
        "video_combine": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["vae_decode", 0],
                "frame_rate": cfg["fps"],
                "loop_count": 0,
                "filename_prefix": cfg["filename_prefix"],
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    return workflow


def queue_prompt(server_address: str, workflow: dict, client_id: str) -> str:
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"http://{server_address}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(server_address: str, client_id: str, prompt_id: str, timeout: int = 1800):
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}", timeout=timeout)
    start = time.time()
    try:
        while True:
            if time.time() - start > timeout:
                raise TimeoutError("ComfyUI 작업 대기 시간 초과")
            message = ws.recv()
            if not isinstance(message, str):
                continue  # 바이너리 프리뷰 프레임은 무시
            data = json.loads(message)
            if data.get("type") == "executing":
                d = data["data"]
                if d.get("prompt_id") == prompt_id and d.get("node") is None:
                    break  # 해당 prompt의 실행이 끝남
            elif data.get("type") == "execution_error" and data["data"].get("prompt_id") == prompt_id:
                raise RuntimeError(f"ComfyUI 실행 오류: {data['data']}")
    finally:
        ws.close()


def download_video(server_address: str, prompt_id: str, output_path: str):
    history = requests.get(f"http://{server_address}/history/{prompt_id}", timeout=30).json()
    outputs = history[prompt_id]["outputs"]

    video_info = None
    for node_output in outputs.values():
        if "gifs" in node_output:  # VHS_VideoCombine은 mp4도 이 키에 담아 반환
            video_info = node_output["gifs"][0]
            break
    if video_info is None:
        raise RuntimeError("출력에서 비디오 파일을 찾지 못했습니다. history 응답을 확인하세요:\n" + json.dumps(outputs, indent=2, ensure_ascii=False))

    params = {
        "filename": video_info["filename"],
        "subfolder": video_info.get("subfolder", ""),
        "type": video_info.get("type", "output"),
    }
    resp = requests.get(f"http://{server_address}/view", params=params, timeout=120)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(resp.content)


def main():
    parser = argparse.ArgumentParser(description="ComfyUI + AnimateDiff로 이미지 -> 비디오 생성")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--output", default="output.mp4", help="저장할 비디오 파일 경로")
    parser.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI 서버 주소 (host:port)")
    parser.add_argument("--checkpoint", default="v1-5-pruned-emaonly.safetensors", help="체크포인트 파일명")
    parser.add_argument("--motion-module", default="mm_sd_v15_v2.ckpt", help="AnimateDiff 모션 모듈 파일명")
    parser.add_argument("--prompt", default="best quality, smooth motion")
    parser.add_argument("--negative-prompt", default="worst quality, blurry, distorted, static")
    parser.add_argument("--frames", type=int, default=16, help="생성할 프레임 수")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--denoise", type=float, default=0.65, help="1.0에 가까울수록 원본 이미지에서 멀어짐")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--filename-prefix", default="animatediff_img2vid")
    args = parser.parse_args()

    cfg = {
        "checkpoint": args.checkpoint,
        "motion_module": args.motion_module,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "denoise": args.denoise,
        "seed": args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1),
        "filename_prefix": args.filename_prefix,
    }

    client_id = str(uuid.uuid4())

    print(f"[1/4] 이미지 업로드 중: {args.image}")
    uploaded_name = upload_image(args.server, args.image)

    print("[2/4] 워크플로우 구성 및 큐 등록 중...")
    workflow = build_workflow(cfg, uploaded_name)
    prompt_id = queue_prompt(args.server, workflow, client_id)
    print(f"      prompt_id = {prompt_id}, seed = {cfg['seed']}")

    print("[3/4] ComfyUI 실행 대기 중 (완료될 때까지 대기)...")
    wait_for_completion(args.server, client_id, prompt_id)

    print(f"[4/4] 결과 비디오 다운로드 중 -> {args.output}")
    download_video(args.server, prompt_id, args.output)

    print("완료:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
