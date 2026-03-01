import cv2
import numpy as np

img1 = cv2.imread("basic_ps.jpg")
img2 = cv2.imread("day.jpg")

def test(i1, i2):
    print(i1[30][30], i2[30][30], i1[30][30]-i2[30][30])

test(img1, img2)
if img1.shape != img2.shape:
    raise ValueError("Images must have the same dimensions")

img1 = img1.astype(np.float32)

img2 = img2.astype(np.float32)
test(img1, img2)

mse_b = np.mean((img1[:,:,0] - img2[:,:,0]) ** 2)
mse_g = np.mean((img1[:,:,1] - img2[:,:,1]) ** 2)
mse_r = np.mean((img1[:,:,2] - img2[:,:,2]) ** 2)

mse_avg = (mse_r + mse_g + mse_b) / 3

print("MSE Blue:", mse_b)
print("MSE Green:", mse_g)
print("MSE Red:", mse_r)
print("Final Channel-wise MSE:", mse_avg)