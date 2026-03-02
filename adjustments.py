import cv2
import numpy as np

def adjust_exposure(img):
    img = 16*np.sqrt(img)
    return img

def remove_red_lights(img):
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red ranges (red wraps around HSV hue scale)
    lower_red1 = np.array([0, 0, 230])
    upper_red1 = np.array([180, 255, 255])

    # Create masks
    mask = cv2.inRange(hsv, lower_red1, upper_red1)

    # Convert to 8-bit color
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Replace bright pixels with nearby values
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # Convert back to float
    img = img.astype(np.float32)
    return img

def adjust_white_balance(img, mask):
    # Compute average per channel
    avg_b = np.mean(img[:, :, 0][mask>0])
    avg_g = np.mean(img[:, :, 1][mask>0])
    avg_r = np.mean(img[:, :, 2][mask>0])

    # Compute scale factors
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    # Apply scaling
    img[:, :, 0] *= scale_b
    img[:, :, 1] *= scale_g
    img[:, :, 2] *= scale_r

    # Clip values to [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def denoise_sky(img):
    # img[400:, :, :] = cv2.fastNlMeansDenoisingColored(img[400:, :, :], None, 0, 4)
    img[:400, :, :] = cv2.fastNlMeansDenoisingColored(img[:400, :, :], None, 10, 10)
    return img

def denoise(img, sky_mask):
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10)
    img[~sky_mask>0] = denoised[~sky_mask>0]
    return img

def denoise_final(img, sky_mask):
    # denoised = cv2.fastNlMeansDenoisingColored(img, None, 0, 0)
    # img[~sky_mask>0] = denoised[~sky_mask>0]
    return img

