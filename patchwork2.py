import cv2
import numpy as np
import os

global_pairs_per_bit = 500

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits) - len(bits) % 8, 8)]
    return ''.join(chars)

def embed_patchwork_message(image_path, message, alpha=5, pairs_per_bit=global_pairs_per_bit, seed=42):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float64)
    if image is None:
        raise ValueError("Image not found or invalid format.")

    height, width, channels = image.shape
    np.random.seed(seed)

    # Convert message to bitstream and prepend 16-bit header for length
    bitstream = text_to_bits(message)
    length = len(bitstream)
    length_bits = format(length, '016b')
    full_bitstream = length_bits + bitstream

    coords_used = set()
    all_pairs = []

    # For each bit, generate random pixel pairs
    for bit in full_bitstream:
        bit_pairs = []
        while len(bit_pairs) < pairs_per_bit:
            y1, x1 = np.random.randint(0, height), np.random.randint(0, width)
            y2, x2 = np.random.randint(0, height), np.random.randint(0, width)

            if (x1, y1, x2, y2) not in coords_used and (x1, y1) != (x2, y2):
                bit_pairs.append((y1, x1, y2, x2))
                coords_used.add((x1, y1, x2, y2))

        all_pairs.append((bit, bit_pairs))

    # Embed the bits
    for bit, pairs in all_pairs:
        for y1, x1, y2, x2 in pairs:
            if bit == '1':
                image[y1, x1] = np.clip(image[y1, x1] + alpha, 0, 255)
                image[y2, x2] = np.clip(image[y2, x2] - alpha, 0, 255)
            else:
                image[y1, x1] = np.clip(image[y1, x1] - alpha, 0, 255)
                image[y2, x2] = np.clip(image[y2, x2] + alpha, 0, 255)

    return image.astype(np.uint8)

def extract_patchwork_message(image_path, alpha=5, pairs_per_bit=global_pairs_per_bit, seed=42):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float64)
    if image is None:
        raise ValueError("Stego image not found.")

    height, width, channels = image.shape
    np.random.seed(seed)

    coords_used = set()
    extracted_bits = ""

    # First extract 16 bits to get message length
    length_bits = ""
    while len(length_bits) < 16:
        bit_pairs = []
        while len(bit_pairs) < pairs_per_bit:
            y1, x1 = np.random.randint(0, height), np.random.randint(0, width)
            y2, x2 = np.random.randint(0, height), np.random.randint(0, width)

            if (x1, y1, x2, y2) not in coords_used and (x1, y1) != (x2, y2):
                bit_pairs.append((y1, x1, y2, x2))
                coords_used.add((x1, y1, x2, y2))

        diffs = [np.sum(image[y1, x1] - image[y2, x2]) for (y1, x1, y2, x2) in bit_pairs]
        S = np.sum(diffs)
        bit = '1' if S > 0 else '0'
        length_bits += bit

    try:
        message_length = int(length_bits, 2)
    except:
        raise ValueError("Corrupted header. Cannot decode message length.")

    # Now extract the message bits
    while len(extracted_bits) < message_length:
        bit_pairs = []
        while len(bit_pairs) < pairs_per_bit:
            y1, x1 = np.random.randint(0, height), np.random.randint(0, width)
            y2, x2 = np.random.randint(0, height), np.random.randint(0, width)

            if (x1, y1, x2, y2) not in coords_used and (x1, y1) != (x2, y2):
                bit_pairs.append((y1, x1, y2, x2))
                coords_used.add((x1, y1, x2, y2))

        diffs = [np.sum(image[y1, x1] - image[y2, x2]) for (y1, x1, y2, x2) in bit_pairs]
        S = np.sum(diffs)
        bit = '1' if S > 0 else '0'
        extracted_bits += bit

    message = bits_to_text(extracted_bits[:message_length])
    return message


def embed_watermark(image_path, message, output_dir=None, alpha=5, pairs_per_bit=global_pairs_per_bit, seed=42):
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")

    watermarked_image = embed_patchwork_message(
        image_path, message, alpha, pairs_per_bit, seed
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir or os.path.dirname(image_path), "patchwork_stego.png")
    cv2.imwrite(output_path, watermarked_image)
    return output_path

def extract_watermark(image_path, pairs_per_bit=global_pairs_per_bit, seed=42):
    watermarked = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if watermarked is None:
        raise ValueError("Stego image could not be read")

    message = extract_patchwork_message(
        image_path, alpha=5, pairs_per_bit=pairs_per_bit, seed=seed
    )

    # Optional sanitization
    printable_chars = ''.join(char for char in message if 32 <= ord(char) <= 126)
    return printable_chars

