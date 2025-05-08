
import cv2
import numpy as np
import os

def text_to_bits(text):
    return ''.join(format(ord(char), '08b') for char in text)

def bits_to_text(bits):
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

def embed_patchwork_message(image_path, message, alpha=5, pairs_per_bit=500, seed=42):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    height, width = image.shape
    np.random.seed(seed)

    bitstream = ''.join(f'{ord(c):08b}' for c in message)
    total_pairs = len(bitstream) * pairs_per_bit
    coords = set()
    pairs = []

    for bit in bitstream:
        bit_pairs = []
        while len(bit_pairs) < pairs_per_bit:
            x1, y1 = np.random.randint(0, height), np.random.randint(0, width)
            x2, y2 = np.random.randint(0, height), np.random.randint(0, width)
            if (x1, y1, x2, y2) not in coords and (x1, y1) != (x2, y2):
                bit_pairs.append((x1, y1, x2, y2))
                coords.add((x1, y1, x2, y2))
        pairs.append(bit_pairs)

    for bit, bit_pairs in zip(bitstream, pairs):
        for x1, y1, x2, y2 in bit_pairs:
            if bit == '1':
                image[x1, y1] = np.clip(image[x1, y1] + alpha, 0, 255)
                image[x2, y2] = np.clip(image[x2, y2] - alpha, 0, 255)
            else:
                image[x1, y1] = np.clip(image[x1, y1] - alpha, 0, 255)
                image[x2, y2] = np.clip(image[x2, y2] + alpha, 0, 255)

    return image.astype(np.uint8), len(bitstream), pairs_per_bit

def extract_patchwork_message(watermarked_path, message_length, alpha=5, pairs_per_bit=500, seed=42):
    watermarked = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    height, width = watermarked.shape
    np.random.seed(seed)

    bitstream = ''
    for _ in range(message_length):
        diffs = []
        for _ in range(pairs_per_bit):
            x1, y1 = np.random.randint(0, height), np.random.randint(0, width)
            x2, y2 = np.random.randint(0, height), np.random.randint(0, width)
            if (x1, y1) != (x2, y2):
                diff = watermarked[x1, y1] - watermarked[x2, y2]
                diffs.append(diff)

        S = np.sum(diffs)
        bit = '1' if S > 0 else '0'
        bitstream += bit

    message = ''.join([chr(int(bitstream[i:i+8], 2)) for i in range(0, len(bitstream), 8)])
    return message

def embed_watermark(image_path, message, output_dir=None, alpha=5, pairs_per_bit=500, seed=42):
    # Check if image exists
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
    
    # Embed the message using the patchwork algorithm
    watermarked_image, message_length, ppb = embed_patchwork_message(
        image_path, message, alpha, pairs_per_bit, seed
    )
    
    # Prepare output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir or os.path.dirname(image_path), "patchwork_stego.png")
    
    # Save watermarked image
    cv2.imwrite(output_path, watermarked_image)
    
    return output_path

def extract_watermark(image_path, message_length=500, pairs_per_bit=500, seed=42):
    # Load watermarked image
    watermarked = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if watermarked is None:
        raise ValueError("Stego image could not be read")
    
    # If message_length is not provided, use a default or try to estimate
    if message_length is None:
        # Default to a reasonable length (e.g., 100 characters = 800 bits)
        message_length = 800
        print("[WARNING] No message length provided, using default length of 100 characters")
    
    # Extract the message using the patchwork algorithm
    message = extract_patchwork_message(
        image_path, message_length, alpha=5, pairs_per_bit=pairs_per_bit, seed=seed
    )
    
    # Try to find the end of the actual message (e.g., by looking for null terminators or invalid chars)
    # This is a simple approach - you might want to implement a more sophisticated method
    printable_chars = ''.join(char for char in message if 32 <= ord(char) <= 126)
    
    return printable_chars