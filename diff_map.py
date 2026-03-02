import cv2
import numpy as np
from pathlib import Path

def generate_diff_maps(im1, im2, output_dir="outputs"):
    assert im1.shape == im2.shape

# get mse for each sq
    mse_b = (im1[:,:,0] - im2[:,:,0]) ** 2
    mse_g = (im1[:,:,1] - im2[:,:,1]) ** 2
    mse_r = (im1[:,:,2] - im2[:,:,2]) ** 2

# Normalize for visualization
    def normalize(img):
        norm = img / 256
        return norm

# Normalize the images
    sq_r_vis = normalize(mse_r)
    sq_g_vis = normalize(mse_g)
    sq_b_vis = normalize(mse_b)

# save grayscale images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_dir/"mse_red.png", sq_r_vis)
    cv2.imwrite(output_dir/"mse_green.png", sq_g_vis)
    cv2.imwrite(output_dir/"mse_blue.png", sq_b_vis)

if __name__ == "__main__":
    night = cv2.imread("images/basic_ps.jpg").astype(np.float32)
    day = cv2.imread("images/day.jpg").astype(np.float32)
    generate_diff_maps(night, day)
