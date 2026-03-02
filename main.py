import cv2
import numpy as np
from diff_map import generate_diff_maps
from sky_detection import detect_skyline
from adjustments import *

def enhance_image(im):
    # Convert image to float for more precision in adjustments
    im = im.astype(np.float32)
    im = adjust_exposure(im)
    im = remove_red_lights(im)
    im = adjust_white_balance(im)
    return im

def calc_mse(im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Images must have the same dimensions")

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    mse_b = np.mean((im1[:,:,0] - im2[:,:,0]) ** 2)
    mse_g = np.mean((im1[:,:,1] - im2[:,:,1]) ** 2)
    mse_r = np.mean((im1[:,:,2] - im2[:,:,2]) ** 2)

    mse_avg = (mse_r + mse_g + mse_b) / 3

    print("MSE Blue:", mse_b)
    print("MSE Green:", mse_g)
    print("MSE Red:", mse_r)
    print("Final Channel-wise MSE:", mse_avg)


if __name__ == "__main__":

    # Import Image
    img = cv2.imread("images/night.jpg")

    # Generate sky mask
    _, sky_mask, _ = detect_skyline(
        img,
        gamma=1.5,
        denoise_h=15,
        top_bias=0.2,
        smoothness=2.0,
        max_jump=20
    )

    # Denoise
    img = denoise(img)
    cv2.imwrite("outputs/denoise.png", img)

    # Apply enhancements
    im_enhance = enhance_image(img)

    # Clip to 8-bit map
    im_enhance = np.clip(im_enhance, 0, 255).astype(np.uint8)
    im_enhance[sky_mask > 0] = [150, 134, 114]

    cv2.imwrite("outputs/enhance.png", im_enhance)

    # Calculate MSE
    img2 = cv2.imread("images/day.jpg")
    calc_mse(im_enhance, img2)
    generate_diff_maps(im_enhance, img2)
