import cv2
import numpy as np


def embed_watermark(image_path, alpha=5, num_pairs=10000):
    """
    Embeds a watermark using the Patchwork Algorithm.
    Args:
        image_path (str): Path to the input grayscale image.
        alpha (float): Strength of the watermark.
        num_pairs (int): Number of pixel pairs to modify.
    Returns:
        watermarked_image (np.array): The watermarked image array.
        pairs (np.array): Array of pixel coordinate pairs used.
        alpha (float): The alpha value used.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or invalid format: {image_path}")

    height, width = image.shape
    watermarked_image = image.astype(np.float64)

    np.random.seed(42)
    pairs = []
    used_coords = set()

    while len(pairs) < num_pairs:
        x1, y1 = np.random.randint(0, height), np.random.randint(0, width)
        x2, y2 = np.random.randint(0, height), np.random.randint(0, width)
        # Ensure pairs are distinct and not repeated
        if (x1, y1) != (x2, y2) and (x1, y1) not in used_coords and (x2, y2) not in used_coords:
            pairs.append((x1, y1, x2, y2))
            used_coords.add((x1, y1))
            used_coords.add((x2, y2))

    pairs = np.array(pairs)

    for x1, y1, x2, y2 in pairs:
        watermarked_image[x1, y1] = np.clip(watermarked_image[x1, y1] + alpha, 0, 255)
        watermarked_image[x2, y2] = np.clip(watermarked_image[x2, y2] - alpha, 0, 255)

    watermarked_image = watermarked_image.astype(np.uint8)
    return watermarked_image, pairs, alpha


def extract_watermark(original_path, watermarked_path, pairs, alpha):
    """
    Extracts watermark by comparing original and watermarked images using correlation.
    Args:
        original_path (str): Path to the original image.
        watermarked_path (str): Path to the watermarked image.
        pairs (list or np.array): List of (x1, y1, x2, y2) pixel coordinate pairs.
        alpha (float): Alpha value used during embedding.
    Returns:
        detection_result (dict): Includes detection strength and flag.
    """
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)

    if original is None or watermarked is None:
        raise ValueError("One or both images could not be read.")

    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)

    diffs = []
    for x1, y1, x2, y2 in pairs:
        diff = (watermarked[x1, y1] - original[x1, y1]) - (watermarked[x2, y2] - original[x2, y2])
        diffs.append(diff)

    diffs = np.array(diffs)
    detection_strength = np.sum(diffs) / (len(pairs) * alpha)
    watermark_detected = detection_strength > 0.05

    return {
        "detection_strength": detection_strength,
        "watermark_detected": watermark_detected
    }
