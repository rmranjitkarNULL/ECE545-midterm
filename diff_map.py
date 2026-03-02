import cv2
import numpy as np

# Load images
night = cv2.imread("basic_ps.jpg").astype(np.float32)
day = cv2.imread("day.jpg").astype(np.float32)

assert night.shape == day.shape

# get mse for each sq
mse_b = (night[:,:,0] - day[:,:,0]) ** 2
mse_g = (night[:,:,1] - day[:,:,1]) ** 2
mse_r = (night[:,:,2] - day[:,:,2]) ** 2

# Normalize for visualization
def normalize(img):
    norm = img / 256
    return norm
    
# Normalize the images
sq_r_vis = normalize(mse_r)
sq_g_vis = normalize(mse_g)
sq_b_vis = normalize(mse_b)

# save grayscale images
cv2.imwrite("mse_red.png", sq_r_vis)
cv2.imwrite("mse_green.png", sq_g_vis)
cv2.imwrite("mse_blue.png", sq_b_vis)