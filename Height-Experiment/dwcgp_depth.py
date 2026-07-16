"""
Depth-aware DWCGP for roadside vegetation height ranking.

Modifies Verma, Zhang & Stockwell (2018) DWCGP by replacing pixel-space
run-lengths with metric-space run-lengths using a monocular depth model.

Stages:
  1. Monocular depth (Depth Anything V2 Metric Outdoor) -> Z[i,j] meters
  2. Grass refinement inside the verge mask via HSV threshold
  3. Vertical orientation detection via Gabor filter bank (4 ori x 5 scale)
  4. Per-column metric run-length: increment by Z/fy instead of 1
  5. Spatial anchoring: back-project the bottom of each column's longest run
  6. DBSCAN patch clustering -> ranked list

We use Depth Anything V2 Metric Outdoor instead of DepthPro because the HF
DepthPro post-processor produced strongly compressed depth on these fisheye
dashcam frames (sky and hood both ~0.15 m). Depth Anything V2 returns
plausible meters (hood ~7 m, horizon ~80 m). Focal length is not predicted
by Depth Anything; we estimate fy from an assumed ~100 deg HFOV (matches
DepthPro's own predicted FOV on the same images), overridable via CLI.

Usage:
  python dwcgp_depth.py --image <path> --mask <path> --output-dir <dir>
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


# ---------------------------------------------------------------------------
# 1. Monocular metric depth
# ---------------------------------------------------------------------------

_DEPTH_CACHE = {}
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def run_depth(image_pil: Image.Image):
    """Run Depth Anything V2 Metric Outdoor. Returns Z[H,W] meters."""
    from transformers import pipeline

    if "pipe" not in _DEPTH_CACHE:
        print(f"[depth] loading {DEPTH_MODEL_ID} ...", flush=True)
        device = 0 if torch.cuda.is_available() else -1
        _DEPTH_CACHE["pipe"] = pipeline(
            "depth-estimation", model=DEPTH_MODEL_ID, device=device
        )
    pipe = _DEPTH_CACHE["pipe"]

    W, H = image_pil.size
    t0 = time.time()
    out = pipe(image_pil)
    pd = out["predicted_depth"]
    if isinstance(pd, torch.Tensor):
        Z = pd.detach().cpu().numpy().astype(np.float32)
    else:
        Z = np.asarray(pd, dtype=np.float32)
    if Z.shape != (H, W):
        Z = cv2.resize(Z, (W, H), interpolation=cv2.INTER_LINEAR)
    print(f"[depth] done in {time.time()-t0:.2f}s; Z shape={Z.shape} "
          f"pcts(1/50/99)={np.percentile(Z,[1,50,99])}", flush=True)
    return Z


def estimate_focal_px(W: int, hfov_deg: float = 100.0) -> float:
    """fy = fx for square pixels; principal point at image centre."""
    return 0.5 * W / np.tan(np.deg2rad(hfov_deg) / 2.0)


# ---------------------------------------------------------------------------
# 2. Grass refinement inside the verge mask
# ---------------------------------------------------------------------------

def grass_mask_from_verge(rgb: np.ndarray, verge: np.ndarray) -> np.ndarray:
    """HSV-threshold green pixels inside the verge mask."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # OpenCV H is 0..179. Green ~ 35..85 in 0..179 scale.
    green = (H >= 30) & (H <= 90) & (S >= 40) & (V >= 25)
    grass = green & (verge > 0)
    grass = cv2.morphologyEx(
        grass.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)
    return grass


# ---------------------------------------------------------------------------
# 3. Gabor verticality
# ---------------------------------------------------------------------------

def gabor_vertical_mask(gray: np.ndarray, target_width: int = 1024) -> np.ndarray:
    """Return boolean mask: dominant Gabor orientation == 90 degrees (vertical).

    The paper's parameters (kernel 11x11, scales 4-16 px wavelength) assume
    a ~640x480 image where grass blades span ~10 px. On 2700-px-wide frames,
    blades span 30-60 px and the filter bank produces almost no vertical hits.
    We downscale to `target_width`, run the bank, then upscale the boolean
    mask back. Same parameter regime, much better hit rate.
    """
    H0, W0 = gray.shape
    if W0 > target_width:
        scale = target_width / W0
        gray_s = cv2.resize(gray, (target_width, int(H0 * scale)),
                            interpolation=cv2.INTER_AREA)
    else:
        gray_s = gray

    gray_f = gray_s.astype(np.float32) / 255.0
    orientations_deg = [0, 45, 90, 135]
    f_max = 0.25
    scales = [f_max / (np.sqrt(2) ** m) for m in range(5)]  # 5 scales
    ksize = 11

    # responses[ori] = max-pooled magnitude across scales
    responses = []
    for theta_deg in orientations_deg:
        theta = np.deg2rad(theta_deg)
        per_scale = []
        for f in scales:
            wavelength = 1.0 / f  # in pixels
            sigma = wavelength * 0.56  # bandwidth ~1 octave
            kern_real = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, wavelength, gamma=0.5,
                psi=0, ktype=cv2.CV_32F,
            )
            kern_imag = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, wavelength, gamma=0.5,
                psi=np.pi / 2, ktype=cv2.CV_32F,
            )
            r = cv2.filter2D(gray_f, cv2.CV_32F, kern_real)
            i = cv2.filter2D(gray_f, cv2.CV_32F, kern_imag)
            per_scale.append(np.sqrt(r * r + i * i))
        responses.append(np.maximum.reduce(per_scale))

    stack = np.stack(responses, axis=0)  # [4, H, W]
    # OpenCV's Gabor theta is the orientation of the normal to the stripes,
    # so theta=0 detects vertical structures and theta=pi/2 detects horizontal.
    # Strict "argmax == vertical" (paper eq. 16) catches almost nothing on these
    # high-res fisheye frames where grass blades are low-contrast or sub-pixel.
    # We relax to "vertical response > horizontal response", i.e. the pixel has
    # more vertical-stem energy than horizontal-edge energy. This still rejects
    # road, sky, foliage tops, and lane lines.
    v_resp = stack[0]   # theta = 0 deg, detects vertical
    h_resp = stack[2]   # theta = 90 deg, detects horizontal
    vertical = v_resp > h_resp

    # Eq. 18 cleanup: flip non-vertical pixel to vertical if its vertical neighbours
    # immediately above and below are both vertical.
    up = np.roll(vertical, 1, axis=0)
    dn = np.roll(vertical, -1, axis=0)
    flip = (~vertical) & up & dn
    vertical = vertical | flip

    # upscale back to original resolution
    if vertical.shape != (H0, W0):
        vertical = cv2.resize(vertical.astype(np.uint8), (W0, H0),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    return vertical


# ---------------------------------------------------------------------------
# 4. Depth-aware per-column run length
# ---------------------------------------------------------------------------

def per_column_metric_height(grass: np.ndarray, vertical: np.ndarray,
                             Z: np.ndarray, fy: float, z_max: float = 30.0):
    """For each column j, find the longest contiguous run of qualifying pixels;
    return arrays of length W:
        l_meters[j]    -> longest metric run length (m)
        bottom_v[j]    -> bottom row index of that run (-1 if none)
    """
    H, W = grass.shape
    qualify = grass & vertical  # [H, W]
    # per-pixel metric vertical extent
    dy = Z / fy  # meters/pixel at depth Z[i,j]
    # mask out pixels with invalid depth or beyond z_max
    valid_z = (Z > 0) & (Z <= z_max)
    qualify = qualify & valid_z

    l_meters = np.zeros(W, dtype=np.float32)
    bottom_v = np.full(W, -1, dtype=np.int32)

    # Iterate column-wise. Vectorising contiguous-runs across columns with
    # variable per-pixel weight is ugly; explicit loop is plenty fast at 2700 cols.
    for j in range(W):
        col_q = qualify[:, j]
        col_dy = dy[:, j]
        cur = 0.0
        cur_start = -1
        best = 0.0
        best_bottom = -1
        for i in range(H):
            if col_q[i]:
                if cur_start < 0:
                    cur_start = i
                cur += float(col_dy[i])
            else:
                if cur > best:
                    best = cur
                    best_bottom = i - 1
                cur = 0.0
                cur_start = -1
        if cur > best:
            best = cur
            best_bottom = H - 1
        l_meters[j] = best
        bottom_v[j] = best_bottom
    return l_meters, bottom_v


# ---------------------------------------------------------------------------
# 5. Back-projection
# ---------------------------------------------------------------------------

def backproject(u: np.ndarray, v: np.ndarray, Z: np.ndarray, fy: float,
                W: int, H: int):
    """Pinhole back-projection with fx = fy and principal point at image centre."""
    cx, cy = W / 2.0, H / 2.0
    Zuv = Z[v, u]
    X = (u - cx) * Zuv / fy
    Y = (v - cy) * Zuv / fy
    return X.astype(np.float32), Y.astype(np.float32), Zuv.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Patch ranking
# ---------------------------------------------------------------------------

def cluster_patches(X: np.ndarray, Z: np.ndarray, heights: np.ndarray,
                    eps: float = 0.5, min_samples: int = 4):
    """Cluster column anchors in (X, Z) world space; return list of dicts
    sorted by 95th-percentile height descending."""
    if len(X) == 0:
        return []
    pts = np.stack([X, Z], axis=1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    patches = []
    for lab in np.unique(labels):
        if lab == -1:
            continue
        sel = labels == lab
        patches.append({
            "label": int(lab),
            "X_mean": float(np.mean(X[sel])),
            "Z_mean": float(np.mean(Z[sel])),
            "height_p95": float(np.percentile(heights[sel], 95)),
            "n_columns": int(sel.sum()),
        })
    patches.sort(key=lambda p: p["height_p95"], reverse=True)
    return patches


# ---------------------------------------------------------------------------
# 7. Visualisation
# ---------------------------------------------------------------------------

def make_figure(rgb: np.ndarray, verge: np.ndarray, Z: np.ndarray,
                grass: np.ndarray, vertical: np.ndarray,
                l_meters: np.ndarray, bottom_v: np.ndarray,
                Xw: np.ndarray, Zw: np.ndarray, heights_anchor: np.ndarray,
                z_max: float, out_path: str):
    H, W = rgb.shape[:2]
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # (a) original + verge overlay
    g = np.zeros_like(rgb); g[..., 1] = 255
    m = (verge > 0)[..., None]
    overlay = np.where(m, (0.55 * rgb + 0.45 * g).astype(np.uint8), rgb)
    ax_a.imshow(overlay); ax_a.set_title("Original + verge mask"); ax_a.axis("off")

    # (b) depth map (clipped to z_max for legibility)
    Zv = np.clip(Z, 0, z_max)
    im = ax_b.imshow(Zv, cmap="turbo")
    ax_b.set_title(f"Predicted depth (m), clipped at {z_max:.0f}")
    ax_b.axis("off"); fig.colorbar(im, ax=ax_b, fraction=0.04)

    # (c) grass mask
    ax_c.imshow(grass.astype(np.uint8) * 255, cmap="Greens")
    ax_c.set_title("Grass mask (HSV inside verge)"); ax_c.axis("off")

    # (d) grass AND vertical mask
    gv = (grass & vertical).astype(np.uint8) * 255
    ax_d.imshow(gv, cmap="gray")
    ax_d.set_title(f"Grass AND vertical (Gabor)  px={int((grass&vertical).sum())}")
    ax_d.axis("off")

    # (e) per-column height bars overlay (drawn on RGB at full res)
    bars = rgb.copy()
    cmap = matplotlib.colormaps["viridis"]
    h_max = 1.5
    bar_w = max(4, W // 400)
    cols_with = np.where(l_meters > 0)[0]
    for j in cols_with:
        h = l_meters[j]; bv = bottom_v[j]
        if bv < 0: continue
        c = (np.array(cmap(min(h / h_max, 1.0))[:3]) * 255).astype(np.uint8)
        bar_h_px = int(np.clip(h / h_max * 200, 8, 280))
        y0 = max(0, bv - bar_h_px); y1 = min(H - 1, bv)
        x0 = max(0, j - bar_w // 2); x1 = min(W, j + bar_w // 2 + 1)
        bars[y0:y1, x0:x1] = c
    ax_e.imshow(bars)
    ax_e.set_title(f"Per-column height bars (viridis 0..{h_max:.1f} m)")
    ax_e.axis("off")

    # (f) top-down scatter of column anchors in world (X, Z)
    if len(Xw):
        sc = ax_f.scatter(Xw, Zw, c=heights_anchor, cmap="viridis",
                          vmin=0, vmax=h_max, s=8)
        fig.colorbar(sc, ax=ax_f, fraction=0.04, label="height (m)")
    ax_f.set_xlabel("X world (m, lateral)")
    ax_f.set_ylabel("Z world (m, forward)")
    ax_f.set_title("Top-down anchors (camera at origin)")
    ax_f.axhline(0, color="k", lw=0.5); ax_f.axvline(0, color="k", lw=0.5)
    ax_f.set_aspect("equal", adjustable="datalim")
    ax_f.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--patch-eps", type=float, default=0.5)
    ap.add_argument("--patch-min-samples", type=int, default=4)
    ap.add_argument("--hfov-deg", type=float, default=100.0,
                    help="Assumed horizontal FOV (used to compute fy). "
                         "DepthPro predicted ~100 deg on these dashcams.")
    ap.add_argument("--focal-length", type=float, default=None,
                    help="Override fy (px) directly. If unset, derived from --hfov-deg.")
    ap.add_argument("--z-max", type=float, default=30.0,
                    help="Pixels with Z > z_max are treated as non-qualifying. "
                         "Depth Anything V2 saturates near 80 m, so we restrict "
                         "to where depth is reliably resolved.")
    args = ap.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    out_dir = Path(args.output_dir) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # load
    rgb_pil = Image.open(image_path).convert("RGB")
    rgb = np.array(rgb_pil)
    H, W = rgb.shape[:2]
    mask_pil = Image.open(mask_path).convert("L")
    if mask_pil.size != (W, H):
        mask_pil = mask_pil.resize((W, H), Image.NEAREST)
    verge = (np.array(mask_pil) > 127).astype(np.uint8)

    print(f"[input] {image_path.name}  size=({W}x{H})  verge_px={verge.sum()}",
          flush=True)

    # depth
    Z = run_depth(rgb_pil)
    fy = float(args.focal_length) if args.focal_length is not None \
        else estimate_focal_px(W, args.hfov_deg)
    print(f"[focal] fy = {fy:.1f} px (HFOV {args.hfov_deg} deg, W={W})", flush=True)

    # grass refinement
    grass = grass_mask_from_verge(rgb, verge)
    print(f"[grass] {grass.sum()} px  ({100*grass.sum()/max(verge.sum(),1):.1f}% of verge)",
          flush=True)

    # gabor verticality
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vertical = gabor_vertical_mask(gray)
    print(f"[gabor] vertical px in verge: {(vertical & (verge>0)).sum()}", flush=True)

    # per-column metric heights
    l_meters, bottom_v = per_column_metric_height(grass, vertical, Z, fy,
                                                  z_max=args.z_max)
    nz = l_meters > 0
    print(f"[runs ] cols with l_j>0: {int(nz.sum())} / {W}", flush=True)

    # anchor in world space using bottom pixel of each column's longest run
    cols = np.where(nz)[0].astype(np.int32)
    heights = l_meters[cols]
    bv = bottom_v[cols]
    Xw, Yw, Zw = backproject(cols, bv, Z, fy, W, H)

    patches = cluster_patches(Xw, Zw, heights,
                              eps=args.patch_eps,
                              min_samples=args.patch_min_samples)

    # sanity stats
    verge_bool = verge > 0
    Zv = Z[verge_bool]
    if Zv.size:
        zmin, zmed, zmax = np.percentile(Zv, [1, 50, 99])
    else:
        zmin = zmed = zmax = float("nan")

    # write report
    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        def w(line=""):
            print(line, flush=True)
            f.write(line + "\n")
        w(f"=== {image_path.name} ===")
        w(f"image size                  : {W} x {H}")
        w(f"depth model                 : {DEPTH_MODEL_ID}")
        w(f"focal length fy (assumed)   : {fy:.1f} px  (HFOV {args.hfov_deg} deg)")
        w(f"depth in verge (1/50/99 pct): {zmin:.2f} / {zmed:.2f} / {zmax:.2f} m")
        w(f"verge px / grass px         : {int(verge_bool.sum())} / {int(grass.sum())}")
        w(f"vertical px in verge        : {int((vertical & verge_bool).sum())}")
        w(f"cols with non-zero l_j      : {int(nz.sum())} / {W}")
        if heights.size:
            w(f"l_j (m) min/median/max      : "
              f"{heights.min():.3f} / {float(np.median(heights)):.3f} / {heights.max():.3f}")
        w()
        w(f"DBSCAN(eps={args.patch_eps} m, min_samples={args.patch_min_samples})")
        w(f"  -> {len(patches)} patches")
        w()
        w(f"{'Rank':>4} | {'Patch center (X,Z) m':>22} | {'Height (m)':>10} | {'n_cols':>6}")
        w("-" * 60)
        for k, p in enumerate(patches[:30], 1):
            ctr = f"({p['X_mean']:+.2f}, {p['Z_mean']:+.2f})"
            w(f"{k:>4} | {ctr:>22} | {p['height_p95']:>10.3f} | {p['n_columns']:>6}")

    # figure
    fig_path = out_dir / "figure.png"
    make_figure(rgb, verge, Z, grass, vertical, l_meters, bottom_v,
                Xw, Zw, heights, args.z_max, str(fig_path))
    print(f"[out  ] wrote {fig_path}\n[out  ] wrote {report_path}", flush=True)

    # also save per-column data for any later analysis
    np.savez(out_dir / "columns.npz",
             cols=cols, X=Xw, Y=Yw, Z=Zw, height=heights,
             fy=fy, image_size=np.array([W, H]))

    return {
        "image": image_path.name,
        "fy": fy,
        "zmin": float(zmin), "zmed": float(zmed), "zmax": float(zmax),
        "n_cols": int(nz.sum()),
        "h_min": float(heights.min()) if heights.size else 0.0,
        "h_med": float(np.median(heights)) if heights.size else 0.0,
        "h_max": float(heights.max()) if heights.size else 0.0,
        "patches": patches,
    }


if __name__ == "__main__":
    main()
