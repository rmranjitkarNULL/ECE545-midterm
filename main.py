import cv2
import numpy as np
from diff_map import generate_diff_maps
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

    img1 = cv2.imread("images/night.jpg")
    img2 = cv2.imread("images/day.jpg")

    img1 = denoise(img1)
    cv2.imwrite("outputs/denoise.png", img1)

    im_enhance = enhance_image(img1)

    # clip to 8-bit map
    im_enhance = np.clip(im_enhance, 0, 255).astype(np.uint8)

    cv2.imwrite("outputs/enhance.png", im_enhance)
    calc_mse(im_enhance, img2)
    generate_diff_maps(im_enhance, img2)
