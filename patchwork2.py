import cv2
import numpy as np
import os

def text_to_bits(text):
    return ''.join(format(ord(char), '08b') for char in text)

def bits_to_text(bits):
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

def embed_watermark(image_path, message, output_dir=None, alpha=5, pairs_per_bit=100, seed=42):
    """Embed a message using patchwork steganography with random pixel pairs."""
    # Convert message to bits
    bitstream = text_to_bits(message)
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    if image is None:
        raise ValueError(f"Image not found or invalid format: {image_path}")
    
    height, width = image.shape
    np.random.seed(seed)

    # Generate random pixel pairs for each bit
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

    # Embed message bits by modifying pixel pairs
    for bit, bit_pairs in zip(bitstream, pairs):
        for x1, y1, x2, y2 in bit_pairs:
            if bit == '1':
                image[x1, y1] = np.clip(image[x1, y1] + alpha, 0, 255)
                image[x2, y2] = np.clip(image[x2, y2] - alpha, 0, 255)
            else:
                image[x1, y1] = np.clip(image[x1, y1] - alpha, 0, 255)
                image[x2, y2] = np.clip(image[x2, y2] + alpha, 0, 255)

    # Save message length and pairs_per_bit in first few pixels
    # This is needed for extraction
    message_length = len(bitstream)
    
    # Store message length in first pixel (R,G,B channels)
    # We'll use LSB of first 24 pixels to store this info
    length_bits = format(message_length, '024b')
    ppb_bits = format(pairs_per_bit, '024b')
    
    # Convert to uint8 for saving
    watermarked_image = image.astype(np.uint8)
    
    # Prepare output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir or os.path.dirname(image_path), "patchwork_stego.png")
    
    # Save metadata in a separate file
    metadata_path = os.path.join(output_dir or os.path.dirname(image_path), "patchwork_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"{message_length}\n{pairs_per_bit}\n{seed}")
    
    # Save watermarked image
    if not cv2.imwrite(output_path, watermarked_image):
        raise ValueError("Failed to save stego image")
    
    print(f"[INFO] Stego-image saved to: {output_path}")
    print(f"[INFO] Metadata saved to: {metadata_path}")
    return output_path

def extract_watermark(image_path):
    """Extract a message using patchwork steganography."""
    # Load watermarked image
    watermarked = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    if watermarked is None:
        raise ValueError("Stego image could not be read")
    
    # Look for metadata file
    metadata_path = os.path.join(os.path.dirname(image_path), "patchwork_metadata.txt")
    
    try:
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
            message_length = int(lines[0].strip())
            pairs_per_bit = int(lines[1].strip())
            seed = int(lines[2].strip())
    except:
        # Default values if metadata file is missing
        print("[WARNING] Metadata file not found, using default parameters")
        message_length = 100  # Assume short message
        pairs_per_bit = 100   # Default value
        seed = 42            # Default seed
    
    height, width = watermarked.shape
    np.random.seed(seed)
    
    # Extract bits
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
    
    # Convert bits to text
    try:
        message = bits_to_text(bitstream)
        print(f"[INFO] Recovered message: {message}")
        return message
    except:
        raise ValueError("Failed to decode message bits.")

# For testing
if __name__ == "__main__":
    image_path = "test_image.png"
    message = "This is a secret message!"
    output_path = embed_watermark(image_path, message)
    extracted = extract_watermark(output_path)
    print(f"Original: {message}")
    print(f"Extracted: {extracted}")

