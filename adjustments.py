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

def adjust_white_balance(img):

    # Compute average per channel
    avg_b = np.mean(img[400:, :, 0])
    avg_g = np.mean(img[400:, :, 1])
    avg_r = np.mean(img[400:, :, 2])

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

def denoise(img):
    img[:400, :, :] = cv2.fastNlMeansDenoisingColored(img[:400, :, :], None, 10, 10)
    img[400:, :, :] = cv2.fastNlMeansDenoisingColored(img[400:, :, :], None, 0, 4)
    # img = cv2.fastNlMeansDenoisingColored(
    #     img,
    #     None,
    #     h=10,        # Filter strength for luminance
    #     hColor=10,   # Filter strength for color
    #     templateWindowSize=7,
    #     searchWindowSize=21
    # )
    return img
