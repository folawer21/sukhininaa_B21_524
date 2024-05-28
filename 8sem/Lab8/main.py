import os
import numpy as np
from numpy import mean
from math import pow, log, log2, floor
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

working_dir = os.getcwd()
input_path = f'{working_dir}/input'
output_path = f'{working_dir}/output'
WHITE = 255
def pixel_gen(img: Image, func=lambda img, pix, x: pix[x]):
    pix = img.load()
    for row in range(img.size[1]):
        for col in range(img.size[0]):
            pos = (col, row)
            yield pos, func(img, pix, pos)


def AV(matrix):
    """Угловой момент второго порядка"""
    normalized_matrix = matrix / np.sum(matrix)
    av = np.sum(np.square(normalized_matrix))
    return av


def D(matrix):
    """Обратный момент различий"""
    d = np.sum(matrix / (1 + np.square(np.arange(matrix.shape[0]))))
    return d

def image_to_np_array(image_name):
    img_src = Image.open(f'input/{image_name}').convert('RGB')
    return np.array(img_src)


def semitone(img):
    return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 *
            img[:, :, 2]).astype(np.uint8)


def haralik(img_arr, d=2):
    matrix = np.zeros(shape=(256, 256))

    for x in range(d, img_arr.shape[0] - d):
        for y in range(d, img_arr.shape[1] - d):
            matrix[img_arr[x - d, y], img_arr[x, y]] += 1
            matrix[img_arr[x + d, y], img_arr[x, y]] += 1
            matrix[img_arr[x, y - d], img_arr[x, y]] += 1
            matrix[img_arr[x, y + d], img_arr[x, y]] += 1

    for x in range(256):
        m = np.array(matrix[x])
        m[np.where(m == 0)] = 1
        matrix[x] = np.log(m)



    matrix = matrix * 256 / np.max(matrix)
    return matrix
def to_semitone(img_name):
    img = image_to_np_array(img_name)
    return Image.fromarray(semitone(img), 'L')

def transform(img: Image, c=1, f0=0, y=0.5):
    # Convert the input image to an Image object if it's a NumPy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    res_img = img.copy()
    d = ImageDraw.Draw(res_img)
    for pos, pixel in pixel_gen(img):
        p = min(int(WHITE * c * (pixel / WHITE + f0) * y), WHITE)
        d.point(pos, p)
    return res_img

def main():
    img_names = ['imagewall.png']

    for img_name in img_names:
        semitone_img = to_semitone(img_name)
        semitone_img.save(f'{output_path}/semitone/{img_name}')
        semi = np.array(Image.open(f'{output_path}/semitone/{img_name}').convert('L'))

        transformed = transform(semi)
        transformed = np.round(transformed).astype(np.uint8)
        transformed_img = Image.fromarray(transformed, "L")
        transformed_img.save(f'{output_path}/contrasted/{img_name}')

        figure, axis = plt.subplots(2, 1)
        axis[0].hist(x=semi.flatten(), bins=np.arange(0, 255))
        axis[0].title.set_text('Исходное изображение')

        axis[1].hist(x=transformed.flatten(), bins=np.arange(0, 255))
        axis[1].title.set_text('Преобразованное изображение')
        plt.tight_layout()
        plt.savefig(f'{output_path}/histograms/{img_name}')

        matrix = haralik(semi.astype(np.uint8))
        result = Image.fromarray(matrix.astype(np.uint8), "L")
        result.save(f'{output_path}/haralik/{img_name}')

        t_matrix = haralik(transformed.astype(np.uint8))
        t_result = Image.fromarray(t_matrix.astype(np.uint8), "L")
        t_result.save(f'{output_path}/haralik_contrasted/{img_name}')

        print('img_name:', img_name)

        print(f"AV: {AV(matrix)}")
        print(f"AV (contrasted): {AV(t_matrix)}")

        print(f"D: {D(matrix)}")
        print(f"D (contrasted): {D(t_matrix)}")

if __name__ == "__main__":
    main()