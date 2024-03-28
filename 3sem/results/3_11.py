from PIL import Image as pim
from PIL import ImageChops as pic
import numpy as np
import matplotlib.pyplot as plt

def moveToSemitone(old_img):
    old_img_arr = np.array(old_img)
    height = old_img_arr.shape[0]
    width = old_img_arr.shape[1]
    new_img_arr = np.zeros(shape=(height, width))
    for x in range(width):
        for y in range(height):
            pixel = old_img_arr[y, x]
            new_img_arr[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]
    return new_img_arr.astype(np.uint8)

def erosion(image, core):
    height, width = image.shape
    core_height,  core_width = core.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(core_height//2, height-core_height//2):
        for j in range(core_width//2, width-core_width//2):
            min_val = 255
            for m in range(core_height):
                for n in range(core_width):
                    if core[m, n] == 1:
                        min_val = min(min_val, image[i-core_height//2+m, j-core_width//2+n])
            result[i, j] = min_val
    return result

def dilation(image, core):
    height, width = image.shape
    core_height, core_width = core.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(core_height//2, height-core_height//2):
        for j in range(core_width//2, width-core_width//2):
            max_val = 0
            for m in range(core_height):
                for n in range(core_width):
                    if core[m, n] == 1:
                        max_val = max(max_val, image[i-core_height//2+m, j-core_width//2+n])
            result[i, j] = max_val
    return result

def opening(image, core):
    eroded_image = erosion(image, core)
    result_image = dilation(eroded_image, core)
    return result_image

def difference(image_1, image_2):
    difference_image = pic.difference(image_1, image_2)
    result_image = pic.invert(difference_image)
    return result_image

start_image = pim.open("im1.png").convert('RGB')
semitone_image = moveToSemitone(start_image)
core = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.uint8)
opened_image = opening(semitone_image,core)
differenceImage = difference(pim.fromarray(semitone_image.astype(np.uint8),'L').convert("RGB"),
                              pim.fromarray(opened_image.astype(np.uint8), 'L').convert('RGB'))
plt.imsave("img1_semitoned.png",semitone_image,cmap = "gray")
plt.imsave("img1_opened.png",opened_image,cmap = "gray")
plt.imsave("img1_difference.png",differenceImage,cmap = "gray")
