import cv2
import numpy as np
from scipy.signal import savgol_filter

def preprocess_night(img_bgr: np.ndarray,
                     use_clahe: bool = True,
                     gamma: float = 1.4,
                     denoise_h: int = 10) -> np.ndarray:
    """ 
    Returns a preprocessed grayscale image for skyline detection.
    - CLAHE on L-channel helps local contrast
    - Gamma brightening helps separate dark sky/buildings
    - fastNlMeans reduces grain
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image passed to preprocess_night")

    # Work in LAB for contrast enhancement
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        L = clahe.apply(L)

    lab = cv2.merge([L, A, B])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gamma correction (brighten if gamma > 1)
    # Convert to float [0,1], apply, back to uint8
    img_f = img.astype(np.float32) / 255.0
    img_f = np.clip(img_f ** (1.0 / max(gamma, 1e-6)), 0, 1)
    img = (img_f * 255.0).astype(np.uint8)



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise (works decently for night grain)
    # h controls strength (10-20 typical)
    gray = cv2.fastNlMeansDenoising(gray, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)

    # Slight blur to suppress salt-and-pepper edges
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Suppress small bright lights
    kernel = np.ones((7,7), np.uint8)
    gray_closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Replace gray with smoothed version
    gray = gray_closed

    return gray


def compute_energy(gray: np.ndarray) -> np.ndarray:
    """
    Improved skyline energy with thin-edge suppression.
    """

    # --- Multi-scale vertical gradients ---
    gy_small = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gy_large = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=9)

    # Larger kernel suppresses tiny lights
    energy = 1.0 * abs(gy_small) + 1.2 * abs(gy_large)

    # Thin edge suppression
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges to measure thickness
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    # Thin edges = edges that disappear when dilated
    thin_edges = edges.astype(np.float32) - dilated.astype(np.float32) * 0.3

    # Subtract thin-edge penalty from energy
    energy = energy - 0.8 * thin_edges

    # --- Normalize to [0,1] ---
    e_min, e_max = energy.min(), energy.max()
    energy = (energy - e_min) / (e_max - e_min + 1e-6)

    return energy


def find_skyline_path(energy: np.ndarray,
                      top_bias: float = 0.6,
                      smoothness: float = 0.8,
                      max_jump: int = 8,
                      vertical_fraction: float = 0.7) -> np.ndarray:
    """
    Dynamic programming to find best path y[x] from left to right.

    energy: (H,W) float in [0,1], higher is better (likely skyline)
    top_bias: encourages skyline to be toward the top (0..1). Higher = stronger bias.
    smoothness: penalty strength for changing y between columns.
    max_jump: maximum allowed vertical move per column (controls smoothness/robustness).

    Returns:
      y_path: shape (W,), integer y for each x
    """
    H, W = energy.shape
    
    # Restrict vertical search region
    max_row = int(H * vertical_fraction)
    energy = energy[:max_row, :]
    H = max_row

    # Add a prior that prefers smaller y (closer to top),
    # because skyline is usually above most buildings.
    y = np.arange(H, dtype=np.float32).reshape(H, 1)
    prior = 1.0 - (y / max(H - 1, 1))          # 1 at top, 0 at bottom
    score = energy + top_bias * prior

    # DP tables
    dp = np.full((H, W), -np.inf, dtype=np.float32)
    parent = np.full((H, W), -1, dtype=np.int32)

    # Initialize first column
    dp[:, 0] = score[:, 0]

    # Precompute smoothness penalties for jumps
    # penalty(dy) = smoothness * (dy^2) normalized
    jumps = np.arange(-max_jump, max_jump + 1, dtype=np.int32)
    penalties = smoothness * (jumps.astype(np.float32) ** 2) / max(1.0, float(max_jump * max_jump))

    # Forward pass
    for x in range(1, W):
        for j, dy in enumerate(jumps):
            y_from = np.arange(H, dtype=np.int32)
            y_to = y_from + dy
            valid = (y_to >= 0) & (y_to < H)
            y_from_v = y_from[valid]
            y_to_v = y_to[valid]

            cand = dp[y_from_v, x - 1] - penalties[j] + score[y_to_v, x]
            # Relaxation
            better = cand > dp[y_to_v, x]
            dp[y_to_v[better], x] = cand[better]
            parent[y_to_v[better], x] = y_from_v[better]

    # Backtrack from best end
    y_end = int(np.argmax(dp[:, W - 1]))
    y_path = np.zeros(W, dtype=np.int32)
    y_path[W - 1] = y_end
    for x in range(W - 1, 0, -1):
        y_path[x - 1] = parent[y_path[x], x] if parent[y_path[x], x] >= 0 else y_path[x]

    return y_path


def skyline_to_mask(h: int, w: int, y_path: np.ndarray) -> np.ndarray:
    """
    Creates a binary sky mask: sky=255 above skyline, 0 below.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        y = int(np.clip(y_path[x], 0, h - 1))
        mask[:y, x] = 255
    return mask


def detect_skyline(img_bgr: np.ndarray,
                   gamma: float = 1.4,
                   denoise_h: int = 10,
                   top_bias: float = 0.6,
                   smoothness: float = 0.8,
                   max_jump: int = 8):

    gray = preprocess_night(img_bgr, use_clahe=True, gamma=gamma, denoise_h=denoise_h)
    energy = compute_energy(gray)

    y_path = find_skyline_path(
        energy,
        top_bias=top_bias,
        smoothness=smoothness,
        max_jump=max_jump
    )

    y_path = savgol_filter(y_path, 51, 3)
    y_path = np.round(y_path).astype(int)

    sky_mask = skyline_to_mask(gray.shape[0], gray.shape[1], y_path)
    
    # Debug overlay
    overlay = img_bgr.copy()
    for x in range(overlay.shape[1]):
        y = int(np.clip(y_path[x], 0, overlay.shape[0]-1))
        cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    debug = {
        "gray": gray,
        "energy": (energy * 255).astype(np.uint8),
        "overlay": overlay
    }

    return y_path, sky_mask, debug


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out_prefix", default="skyline", help="Output prefix")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    y_path, sky_mask, dbg = detect_skyline(
        img,
        gamma=1.5,
        denoise_h=15,
        top_bias=0.2,
        smoothness=2.0,
        max_jump=20
    )

    cv2.imwrite(f"{args.out_prefix}_mask.png", sky_mask)
    cv2.imwrite(f"{args.out_prefix}_overlay.png", dbg["overlay"])
    cv2.imwrite(f"{args.out_prefix}_energy.png", dbg["energy"])
    cv2.imwrite(f"{args.out_prefix}_gray.png", dbg["gray"])
    print("Wrote:",
          f"{args.out_prefix}_mask.png,",
          f"{args.out_prefix}_overlay.png,",
          f"{args.out_prefix}_energy.png,",
          f"{args.out_prefix}_gray.png")