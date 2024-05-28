import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageOps

hebrew_letters = [
    '\u05D0',  # Aleph
    '\u05D1',  # Bet
    '\u05D2',  # Gimel
    '\u05D3',  # Dalet
    '\u05D4',  # He
    '\u05D5',  # Vav
    '\u05D6',  # Zayin
    '\u05D7',  # Het
    '\u05D8',  # Tet
    '\u05D9',  # Yod
    '\u05DA',  # Final Kaf
    '\u05DB',  # Kaf
    '\u05DC',  # Lamed
    '\u05DD',  # Final Mem
    '\u05DE',  # Mem
    '\u05DF',  # Final Nun
    '\u05E0',  # Nun
    '\u05E1',  # Samekh
    '\u05E2',  # Ayin
    '\u05E3',  # Final Pe
    '\u05E4',  # Pe
    '\u05E5',  # Final Tsadi
    '\u05E6',  # Tsadi
    '\u05E7',  # Qof
    '\u05E8',  # Resh
    '\u05E9',  # Shin
    '\u05EA'   # Tav
]

# Создание словаря для быстрого поиска
hebrew_dict = {
    'א': hebrew_letters[0],
    'ב': hebrew_letters[1],
    'ג': hebrew_letters[2],
    'ד': hebrew_letters[3],
    'ה': hebrew_letters[4],
    'ו': hebrew_letters[5],
    'ז': hebrew_letters[6],
    'ח': hebrew_letters[7],
    'ט': hebrew_letters[8],
    'י': hebrew_letters[9],
    'ך': hebrew_letters[10],
    'כ': hebrew_letters[11],
    'ל': hebrew_letters[12],
    'ם': hebrew_letters[13],
    'מ': hebrew_letters[14],
    'ן': hebrew_letters[15],
    'נ': hebrew_letters[16],
    'ס': hebrew_letters[17],
    'ע': hebrew_letters[18],
    'ף': hebrew_letters[19],
    'פ': hebrew_letters[20],
    'ץ': hebrew_letters[21],
    'צ': hebrew_letters[22],
    'ק': hebrew_letters[23],
    'ר': hebrew_letters[24],
    'ש': hebrew_letters[25],
    'ת': hebrew_letters[26],
}

# Исходная фраза
phrase = "השאירי את פנייך בשמש ולא תוכלי לראות כל צל"

# Преобразование фразы
PHRASE = ''.join([hebrew_dict[char] if char in hebrew_dict else char for char in phrase])

PHRASE="השאיריאתפנייךבשמשולאתוכלילראותכלצל"
WHITE=255

FONT=ImageFont.truetype("input/ArialHB.ttc",52)
THRESHOLD=75

def create_phrase_profiles(img: np.array):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != WHITE] = 1  # Assuming white pixel value is 255

    plt.bar(
        x=np.arange(start=1, stop=img_b.shape[1] + 1).astype(int),
        height=np.sum(img_b, axis=0),
        width=0.9
    )
    plt.savefig(f'output/phrase_profile/x.png')
    plt.clf()

    plt.barh(
        y=np.arange(start=1, stop=img_b.shape[0] + 1).astype(int),
        width=np.sum(img_b, axis=1),
        height=0.9
    )
    plt.savefig(f'output/phrase_profile/y.png')
    plt.clf()


def _simple_binarization(img, threshold=THRESHOLD):
    new_image = np.zeros(shape=img.shape)
    new_image[img > threshold] = WHITE
    return new_image.astype(np.uint8)


def generate_phrase_image():
    space_len = 5
    phrase_width = sum(FONT.getsize(char)[0] for char in PHRASE) + space_len * (len(PHRASE) - 1)

    height = max(FONT.getsize(char)[1] for char in PHRASE)

    img = Image.new("L", (phrase_width, height), color="white")
    draw = ImageDraw.Draw(img)

    current_x = 0
    for letter in PHRASE:
        width, letter_height = FONT.getsize(letter)
        draw.text((current_x, height - letter_height), letter, "black", font=FONT)
        current_x += width + space_len

    img = Image.fromarray(_simple_binarization(np.array(img)))
    img.save("output/original_phrase.bmp")

    np_img = np.array(img)
    create_phrase_profiles(np_img)
    ImageOps.invert(img).save("output/inverted_phrase.bmp")
    return np_img


def segment_letters(img):
    # Find stard and end of each letter
    profile = np.sum(img == 0, axis=0)

    in_letter = False
    letter_bounds = []

    for i in range(len(profile)):
        if profile[i] > 0:
            if not in_letter:
                in_letter = True
                start = i
        else:
            if in_letter:
                in_letter = False
                end = i
                letter_bounds.append((start - 1, end))

    if in_letter:
        letter_bounds.append((start, len(profile)))

    return letter_bounds

def draw_bounding_boxes(img, bounds):
    # Draw rectangles for each letter border
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)

    for start, end in bounds:
        left, right = start, end
        top, bottom = 0, img.shape[0]
        draw.rectangle([left, top, right, bottom], outline="red")

    image.save("output/segmented_phrase.bmp")
if __name__ == "__main__":
    img = generate_phrase_image()
    bounds = segment_letters(img)
    draw_bounding_boxes(img, bounds)