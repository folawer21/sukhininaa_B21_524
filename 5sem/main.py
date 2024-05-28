import csv
import os
from math import ceil
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from matplotlib import pyplot as plt

hebrew_letters = [
    '\u05D0', '\u05D1', '\u05D2',  '\u05D3', '\u05D4', '\u05D5', '\u05D6','\u05D7','\u05D8','\u05D9',
    '\u05DA', '\u05DB', '\u05DC', '\u05DD',  '\u05DE',  '\u05DF', '\u05E0', '\u05E1', '\u05E2',
    '\u05E3', '\u05E4',  '\u05E5',  '\u05E6', '\u05E7', '\u05E8', '\u05E9','\u05EA'
]

TYPOGRAPHIC_UNIT_SIZE = 52
THRESHOLD = 75
TTF_FILE = "input/ArialHB.ttc"

WHITE = 255

def _simple_binarization(img, threshold=THRESHOLD):
    semitoned = (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]).astype(np.uint8)
    new_image = np.zeros(shape=semitoned.shape)
    new_image[semitoned > threshold] = WHITE
    return new_image.astype(np.uint8)


def generate_letters(sin_letters):
    font = ImageFont.truetype(TTF_FILE, TYPOGRAPHIC_UNIT_SIZE)
    os.makedirs("output/letters", exist_ok=True)
    os.makedirs("output/inverse_letters", exist_ok=True)

    for i in range(len(sin_letters)):
        letter = sin_letters[i]

        width, height = font.getsize(letter)
        img = Image.new(mode="RGB", size=(ceil(width), ceil(height)), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), letter, "black", font=font)

        img = Image.fromarray(_simple_binarization(np.array(img), THRESHOLD), 'L')
        img.save(f"output/letters/{i + 1}.png")

        ImageOps.invert(img).save(f"output/inverse_letters/{i + 1}.png")


def calculate_features(img):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != WHITE] = 1

    # Calculate quadrant weights and relative weights

    (h, w) = img_b.shape
    h_half, w_half = h // 2, w // 2
    quadrants = {
        'top_left': img_b[:h_half, :w_half],
        'top_right': img_b[:h_half, w_half:],
        'bottom_left': img_b[h_half:, :w_half],
        'bottom_right': img_b[h_half:, w_half:]
    }
    weights = {k: np.sum(v) for k, v in quadrants.items()}
    rel_weights = {k: v / (h_half * w_half) for k, v in weights.items()}

    # Calculate center of mass
    total_pixels = np.sum(img_b)  # count black at least...
    y_indices, x_indices = np.indices(img_b.shape)
    y_center_of_mass = np.sum(y_indices * img_b) / total_pixels
    x_center_of_mass = np.sum(x_indices * img_b) / total_pixels
    center_of_mass = (x_center_of_mass, y_center_of_mass)

    # Calculate normalized center of mass
    normalized_center_of_mass = (x_center_of_mass / (w - 1), y_center_of_mass / (h - 1))

    # Calculate inertia
    inertia_x = np.sum((y_indices - y_center_of_mass) ** 2 * img_b) / total_pixels
    normalized_inertia_x = inertia_x / h ** 2
    inertia_y = np.sum((x_indices - x_center_of_mass) ** 2 * img_b) / total_pixels
    normalized_inertia_y = inertia_y / w ** 2

    return {
        'weight': total_pixels,
        'weights': weights,
        'rel_weights': rel_weights,
        'center_of_mass': center_of_mass,
        'normalized_center_of_mass': normalized_center_of_mass,
        'inertia': (inertia_x, inertia_y),
        'normalized_inertia': (normalized_inertia_x, normalized_inertia_y)
    }


def create_features(sin_letters):
    with open('output/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['weight', 'weights', 'rel_weights', 'center_of_mass',
                      'normalized_center_of_mass','inertia', 'normalized_inertia']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(sin_letters)):
            img_src = np.array(Image.open(f'output/letters/{i + 1}.png').convert('L'))
            features = calculate_features(img_src)
            writer.writerow(features)


def create_profiles(sin_letters):
    os.makedirs("output/profiles/x", exist_ok=True)
    os.makedirs("output/profiles/y", exist_ok=True)

    for i in range(len(sin_letters)):
        img = np.array(Image.open(f'output/letters/{i + 1}.png').convert('L'))
        img_b = np.zeros(img.shape, dtype=int)
        img_b[img != WHITE] = 1  # Assuming white pixel value is 255

        plt.bar(
            x=np.arange(start=1, stop=img_b.shape[1] + 1).astype(int),
            height=np.sum(img_b, axis=0),
            width=0.9
        )
        plt.ylim(0, TYPOGRAPHIC_UNIT_SIZE)
        plt.xlim(0, 55)
        plt.savefig(f'output/profiles/x/{i + 1}.png')
        plt.clf()
        plt.barh(
            y=np.arange(start=1, stop=img_b.shape[0] + 1).astype(int),
            width=np.sum(img_b, axis=1),
            height=0.9
        )
        plt.ylim(TYPOGRAPHIC_UNIT_SIZE, 0)
        plt.xlim(0, 55)
        plt.savefig(f'output/profiles/y/{i + 1}.png')
        plt.clf()
if __name__ == "__main__":
    generate_letters(hebrew_letters)
    create_features(hebrew_letters)
    create_profiles(hebrew_letters)