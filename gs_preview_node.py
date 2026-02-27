"""
ComfyUI Custom Node: 3D Gaussian Splatting Preview

Pipeline:
  1. SIFT/ORB feature matching between 2 input images
  2. Essential matrix → relative camera pose (R, t)
  3. Point triangulation → sparse 3D point cloud
  4. Gaussian initialization at each 3D point
  5. Novel-view rendering via depth-sorted Gaussian splatting

Note: Full 3DGS training requires many images; with 2 images we produce
a sparse Gaussian scene and render it from a novel viewpoint.
"""

import math

import numpy as np
import torch


class GaussianSplattingPreviewNode:
    """
    Input  : 2 images (ComfyUI IMAGE tensor, shape B×H×W×C, float32 [0,1])
    Output : rendered preview image (same tensor format)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
            "optional": {
                "render_width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                }),
                "render_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                }),
                "max_gaussians": ("INT", {
                    "default": 1000, "min": 100, "max": 5000, "step": 100,
                    "tooltip": "Triangulated points used as Gaussian centers",
                }),
                "gaussian_scale": ("FLOAT", {
                    "default": 0.025, "min": 0.001, "max": 0.3, "step": 0.001,
                    "tooltip": "Gaussian size relative to focal length",
                }),
                "novel_view_yaw": ("FLOAT", {
                    "default": 15.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Horizontal rotation of novel viewpoint (degrees)",
                }),
                "novel_view_pitch": ("FLOAT", {
                    "default": -5.0, "min": -45.0, "max": 45.0, "step": 1.0,
                    "tooltip": "Vertical tilt of novel viewpoint (degrees)",
                }),
                "background_color": (["dark", "white", "black", "transparent"], {
                    "default": "dark",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_preview",)
    FUNCTION = "render_preview"
    CATEGORY = "3D/Gaussian Splatting"
    OUTPUT_NODE = False

    # ──────────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────────

    def render_preview(
        self,
        image1, image2,
        render_width=512, render_height=512,
        max_gaussians=1000, gaussian_scale=0.025,
        novel_view_yaw=15.0, novel_view_pitch=-5.0,
        background_color="dark",
    ):
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "[3DGS] OpenCV not found. Install with: pip install opencv-python"
            )

        print("[3DGS] Starting Gaussian Splatting preview pipeline…")

        # ComfyUI IMAGE tensor: (B, H, W, C) float32 [0,1] → uint8 numpy
        img1 = (image1[0].cpu().numpy() * 255).astype(np.uint8)
        img2 = (image2[0].cpu().numpy() * 255).astype(np.uint8)

        # ── 3D reconstruction ──────────────────────────────────────────────
        result = self._reconstruct(img1, img2, max_gaussians)

        if result is None:
            print("[3DGS] Reconstruction failed — using side-by-side fallback.")
            return self._fallback(image1, image2, render_width, render_height)

        pts3d, colors = result
        print(f"[3DGS] Reconstructed {len(pts3d)} Gaussians. Rendering…")

        # ── Gaussian rendering ─────────────────────────────────────────────
        rendered = self._render(
            pts3d, colors,
            render_width, render_height,
            gaussian_scale, novel_view_yaw, novel_view_pitch,
            background_color,
        )

        out = torch.from_numpy(rendered.astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 – Sparse 3D Reconstruction
    # ──────────────────────────────────────────────────────────────────────────

    def _reconstruct(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray, max_pts: int):
        import cv2

        h, w = img1_rgb.shape[:2]
        g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

        # Feature detector — SIFT preferred, ORB fallback
        try:
            det = cv2.SIFT_create(nfeatures=max_pts * 4)
            norm = cv2.NORM_L2
        except Exception:
            det = cv2.ORB_create(nfeatures=max_pts * 4)
            norm = cv2.NORM_HAMMING

        kp1, d1 = det.detectAndCompute(g1, None)
        kp2, d2 = det.detectAndCompute(g2, None)

        if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
            print("[3DGS] Not enough keypoints detected.")
            return None

        # Ratio-test matching
        bf = cv2.BFMatcher(norm, crossCheck=False)
        try:
            raw = bf.knnMatch(d1, d2, k=2)
            good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        except Exception:
            return None

        if len(good) < 8:
            print(f"[3DGS] Too few good matches: {len(good)}")
            return None

        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Estimated camera intrinsics
        f = max(h, w) * 1.2
        K = np.float64([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

        # Essential matrix + pose recovery
        E, mask_e = cv2.findEssentialMat(
            p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            print("[3DGS] Essential matrix estimation failed.")
            return None

        p1 = p1[mask_e.ravel() == 1]
        p2 = p2[mask_e.ravel() == 1]

        _, R, t, mask_p = cv2.recoverPose(E, p1, p2, K)
        p1 = p1[mask_p.ravel() > 0]
        p2 = p2[mask_p.ravel() > 0]

        if len(p1) < 4:
            print("[3DGS] Too few inliers after pose recovery.")
            return None

        # Triangulation
        P1 = K @ np.eye(3, 4)
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        pts3d = (pts4d[:3] / pts4d[3:]).T

        # Validity filter
        ok = np.isfinite(pts3d).all(axis=1) & (pts3d[:, 2] > 0)
        pts3d, p1 = pts3d[ok], p1[ok]
        if len(pts3d) < 4:
            return None

        # Depth outlier removal
        z = pts3d[:, 2]
        zm = np.median(z)
        ok = (z > zm * 0.01) & (z < zm * 100)
        pts3d, p1 = pts3d[ok], p1[ok]
        if len(pts3d) < 4:
            return None

        # Normalize to unit scale centered at origin
        centroid = pts3d.mean(axis=0)
        pts3d -= centroid
        scale = np.percentile(np.linalg.norm(pts3d, axis=1), 90)
        if scale > 1e-8:
            pts3d /= scale

        # Subsample
        if len(pts3d) > max_pts:
            idx = np.random.choice(len(pts3d), max_pts, replace=False)
            pts3d, p1 = pts3d[idx], p1[idx]

        # Sample colors from image1 at feature locations
        px = p1[:, 0].astype(int).clip(0, img1_rgb.shape[1] - 1)
        py = p1[:, 1].astype(int).clip(0, img1_rgb.shape[0] - 1)
        colors = img1_rgb[py, px].astype(np.float32) / 255.0

        print(f"[3DGS] {len(pts3d)} triangulated points (inliers after filtering).")
        return pts3d.astype(np.float32), colors

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 – Gaussian Splatting Render
    # ──────────────────────────────────────────────────────────────────────────

    def _render(
        self,
        pts3d: np.ndarray,
        colors: np.ndarray,
        W: int, H: int,
        gs_scale: float,
        yaw_deg: float, pitch_deg: float,
        bg_mode: str,
    ) -> np.ndarray:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Novel-view camera rotation ──────────────────────────────────────
        yaw   = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        Ry = np.array([
            [ math.cos(yaw), 0, math.sin(yaw)],
            [ 0,             1, 0            ],
            [-math.sin(yaw), 0, math.cos(yaw)],
        ], dtype=np.float32)

        Rx = np.array([
            [1, 0,               0              ],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch),  math.cos(pitch)],
        ], dtype=np.float32)

        R = Rx @ Ry  # combined rotation

        # Camera placed at z=2.5 looking toward the origin
        cam_pos = np.array([0.0, 0.0, 2.5], dtype=np.float32)

        # Transform to camera space
        pts_cam = (pts3d - cam_pos) @ R.T  # (N, 3)

        # Keep only points in front of camera
        front = pts_cam[:, 2] > 0.05
        pts_cam = pts_cam[front]
        col     = colors[front]

        if len(pts_cam) == 0:
            print("[3DGS] No points visible from novel view — returning blank.")
            return np.zeros((H, W, 3), dtype=np.uint8)

        # ── Pinhole projection ──────────────────────────────────────────────
        focal = min(W, H) * 0.9
        z  = pts_cam[:, 2]
        xp = (pts_cam[:, 0] / z) * focal + W / 2.0
        yp = (pts_cam[:, 1] / z) * focal + H / 2.0

        # Screen-space Gaussian sigma (larger = blurrier / bigger blobs)
        sigma = np.clip(focal * gs_scale / z, 1.0, 50.0)

        # ── Sort back-to-front (painter's algorithm) ────────────────────────
        order = np.argsort(z)[::-1]
        xp, yp, sigma, col = xp[order], yp[order], sigma[order], col[order]

        # ── To torch ───────────────────────────────────────────────────────
        xp_t = torch.from_numpy(xp).to(device)
        yp_t = torch.from_numpy(yp).to(device)
        sg_t = torch.from_numpy(sigma).to(device)
        cl_t = torch.from_numpy(col).to(device)     # (N, 3)

        # Pixel coordinate grids (H, W)
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        canvas = torch.zeros(H, W, 3, device=device)
        weight = torch.zeros(H, W,    device=device)

        # ── Batched weighted-additive Gaussian splatting ────────────────────
        #   canvas[h,w] = Σ_i  gauss_i(h,w) * color_i
        #   weight[h,w] = Σ_i  gauss_i(h,w)
        #   rgb[h,w]    = canvas / weight   (weighted average)
        #   alpha[h,w]  = 1 - exp(-weight)  (opacity from accumulation)
        BATCH = 64
        N = len(xp_t)

        for i in range(0, N, BATCH):
            bx = xp_t[i : i + BATCH]      # (B,)
            by = yp_t[i : i + BATCH]
            bs = sg_t[i : i + BATCH]
            bc = cl_t[i : i + BATCH]      # (B, 3)

            dx = gx[:, :, None] - bx[None, None, :]  # (H, W, B)
            dy = gy[:, :, None] - by[None, None, :]

            gauss = torch.exp(
                -(dx ** 2 + dy ** 2) / (2.0 * bs[None, None, :] ** 2)
            )  # (H, W, B)

            # Weighted color accumulation
            canvas += (gauss[:, :, :, None] * bc[None, None, :, :]).sum(dim=2)
            weight += gauss.sum(dim=2)

        # ── Composite over background ───────────────────────────────────────
        alpha = (1.0 - torch.exp(-weight * 2.5)).unsqueeze(-1)   # (H, W, 1)
        w_safe = weight.clamp(min=1e-8).unsqueeze(-1)
        rgb_gs = canvas / w_safe                                  # weighted avg

        bg_map = {
            "dark":        torch.tensor([0.05, 0.05, 0.10], device=device),
            "white":       torch.tensor([1.00, 1.00, 1.00], device=device),
            "black":       torch.tensor([0.00, 0.00, 0.00], device=device),
            "transparent": torch.tensor([0.05, 0.05, 0.10], device=device),
        }
        bg = bg_map.get(bg_mode, bg_map["dark"])

        rgb = rgb_gs * alpha + bg[None, None, :] * (1.0 - alpha)
        rgb = rgb.clamp(0.0, 1.0)

        return (rgb.cpu().numpy() * 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # Fallback – side-by-side display when reconstruction fails
    # ──────────────────────────────────────────────────────────────────────────

    def _fallback(self, image1, image2, W: int, H: int):
        from PIL import Image as PILImage

        img1 = (image1[0].cpu().numpy() * 255).astype(np.uint8)
        img2 = (image2[0].cpu().numpy() * 255).astype(np.uint8)

        p1 = PILImage.fromarray(img1).resize((W // 2, H), PILImage.LANCZOS)
        p2 = PILImage.fromarray(img2).resize((W // 2, H), PILImage.LANCZOS)

        out = PILImage.new("RGB", (W, H), (20, 20, 30))
        out.paste(p1, (0, 0))
        out.paste(p2, (W // 2, 0))

        arr = np.array(out).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)
