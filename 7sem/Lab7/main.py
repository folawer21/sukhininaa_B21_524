import csv
import math
import numpy as np
from PIL import Image
HEBREW_LETTERS_UNICODE = [
    "05D0", "05D1", "05D2", "05D3", "05D4", "05D5", "05D6", "05D7", "05D8", "05D9", "05DA", "05DB", "05DC", "05DD",
    "05DE", "05DF", "05E0", "05E1", "05E2", "05E3", "05E4", "05E5", "05E6", "05E7", "05E8", "05E9", "05EA"
]
HEBREW_LETTERS = [chr(int(letter, 16)) for letter in HEBREW_LETTERS_UNICODE]

WHITE = 255
PHRASE = "השאירי את פנייך בשמש ולא תוכלי לראות כל צל".replace(" ", "")

def calculate_features(img: np.array):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != WHITE] = 1  # Assuming white pixel value is 255

    # Calculate weight
    weight = np.sum(img_b)

    # Calculate center of mass
    y_indices, x_indices = np.indices(img_b.shape)
    y_center_of_mass = np.sum(y_indices * img_b) / weight
    x_center_of_mass = np.sum(x_indices * img_b) / weight

    # Calculate inertia
    inertia_x = np.sum((y_indices - y_center_of_mass) ** 2 * img_b) / weight
    inertia_y = np.sum((x_indices - x_center_of_mass) ** 2 * img_b) / weight

    return weight, x_center_of_mass, y_center_of_mass, inertia_x, inertia_y


def segment_letters(img):
    # Finds the start and end of each letter.
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


def get_alphabet_info() -> dict[chr, tuple]:
    def parse_tuple(string):
        # Converts a string like '(x, y)' to a tuple (x, y).
        a = string.replace("np.float64(","").replace(")","").replace("(","")
        return tuple(map(float, a.split(',')))

    tuples_list = dict()
    with open('input/data.csv', 'r') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            weight = int(row['weight'])
            center_of_mass = parse_tuple(row['center_of_mass'])
            inertia = parse_tuple(row['inertia'])
            tuples_list[HEBREW_LETTERS[i]] = weight, *center_of_mass, *inertia
            i += 1
    return tuples_list


def create_hypothesis(alphabet_info: dict[chr, tuple], target_features):
    def euclidean_distance(feature1, feature2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(feature1, feature2)))

    # Returns a list of tuples (letter, similarity measure) sorted by descending similarity.
    distances = dict()
    for letter, features in alphabet_info.items():
        distance = euclidean_distance(target_features, features)
        distances[letter] = distance

    max_distance = max(distances.values())

    similarities = [(letter, round(1 - distance / max_distance, 2)) for letter, distance in distances.items()]

    return sorted(similarities, key=lambda x: x[1])


def get_phrase_from_hypothesis(img: np.array, bounds) -> str:
    alphabet_info = get_alphabet_info()
    res = []
    for start, end in bounds:
        letter_features = calculate_features(img[:, start: end])
        hypothesis = create_hypothesis(alphabet_info, letter_features)
        best_hypotheses = hypothesis[-1][0]
        res.append(best_hypotheses)
    return "".join(res)


def write_res(recognized_phrase: str):
    max_len = max(len(PHRASE), len(recognized_phrase))
    orig = PHRASE.ljust(max_len)
    detected = recognized_phrase.ljust(max_len)
    with open("output/result.txt", 'w', encoding="utf-8") as f:
        correct_letters = 0
        by_letter = ["has | got | correct"]
        for i in range(max_len):
            is_correct = orig[i] == detected[i]
            by_letter.append(f"{orig[i]}\t{detected[i]}\t{is_correct}")
            correct_letters += int(is_correct)
        f.write("\n".join([
            f"phrase:      {orig}",
            f"detected:    {detected}",
            f"correct:     {math.ceil(correct_letters / max_len * 100)}%\n\n"]))
        f.write("\n".join(by_letter))

if __name__ == "__main__":
    img = np.array(Image.open(f'input/original_phrase.bmp').convert('L'))
    bounds = segment_letters(img)
    recognized_phrase = get_phrase_from_hypothesis(img, bounds)
    write_res(recognized_phrase)