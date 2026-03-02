import cv2
import numpy as np
from diff_map import generate_diff_maps

if __name__ == "__main__":

    img1 = cv2.imread("images/basic_ps.jpg")
    img2 = cv2.imread("images/day.jpg")

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse_b = np.mean((img1[:,:,0] - img2[:,:,0]) ** 2)
    mse_g = np.mean((img1[:,:,1] - img2[:,:,1]) ** 2)
    mse_r = np.mean((img1[:,:,2] - img2[:,:,2]) ** 2)

    mse_avg = (mse_r + mse_g + mse_b) / 3

    print("MSE Blue:", mse_b)
    print("MSE Green:", mse_g)
    print("MSE Red:", mse_r)
    print("Final Channel-wise MSE:", mse_avg)

    generate_diff_maps(img1, img2)
