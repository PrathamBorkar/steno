import cv2
import numpy as np
import os

def text_to_bits(text):
    """Convert text string into a string of bits."""
    bits = ''.join(format(ord(c), '08b') for c in text)
    return bits

def bits_to_text(bits):
    """Convert a string of bits back into text."""
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

def get_range_table():
    """Define the quantization range table"""
    return [
        (0, 15, 2),    # (lower, upper, bits_to_hide)
        (16, 31, 3),
        (32, 255, 4)
    ]

def find_range(diff):
    """Find the appropriate range for the difference"""
    range_table = get_range_table()
    for lower, upper, bits in range_table:
        if lower <= diff <= upper:
            return lower, upper, bits
    return range_table[-1]  # Return last range if diff is too large

def embed_pvd_channel(channel_data, secret_bits, bit_idx):
    """Embed bits into a single channel using PVD."""
    flat_channel = channel_data.flatten()
    i = 0
    while i < len(flat_channel) - 1 and bit_idx < len(secret_bits):
        p1 = flat_channel[i]
        p2 = flat_channel[i+1]
        
        diff = abs(int(p1) - int(p2))
        lower, upper, bits_to_hide = find_range(diff)
        
        if bit_idx + bits_to_hide > len(secret_bits):
            bits_to_hide = len(secret_bits) - bit_idx
        
        if bits_to_hide == 0:
            break
        
        bits = secret_bits[bit_idx:bit_idx+bits_to_hide]
        hidden_value = int(bits, 2)
        
        # Check if adding hidden value exceeds range
        new_diff = lower + hidden_value
        if new_diff > upper:
            # If exceeds, reduce bits to hide
            bits_to_hide -= 1
            if bits_to_hide > 0:
                bits = secret_bits[bit_idx:bit_idx+bits_to_hide]
                hidden_value = int(bits, 2)
                new_diff = lower + hidden_value
            else:
                i += 2
                continue
        
        # Modify pixels based on original relationship
        if p1 > p2:
            flat_channel[i] = p2 + new_diff
            flat_channel[i+1] = p2
        else:
            flat_channel[i+1] = p1 + new_diff
            flat_channel[i] = p1
        
        flat_channel[i] = np.clip(flat_channel[i], 0, 255)
        flat_channel[i+1] = np.clip(flat_channel[i+1], 0, 255)
        
        bit_idx += bits_to_hide
        i += 2
    
    return flat_channel.reshape(channel_data.shape), bit_idx

def embed_pvd(image_path, secret_text, output_dir=None):
    # Load the color image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

    # Split into channels
    b, g, r = cv2.split(image)
    
    secret_bits = text_to_bits(secret_text)
    secret_bits += '1111111111111110'  # End Marker
    bit_idx = 0
    
    # Embed in each channel sequentially
    r_stego, bit_idx = embed_pvd_channel(r, secret_bits, bit_idx)
    g_stego, bit_idx = embed_pvd_channel(g, secret_bits, bit_idx)
    b_stego, bit_idx = embed_pvd_channel(b, secret_bits, bit_idx)
    
    # Merge channels
    stego_image = cv2.merge([b_stego, g_stego, r_stego])
    
    # Generate output path
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Create output filename
    input_filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(input_filename)[0]
    output_path = os.path.join(output_dir, "pvd_stego.png")
    
    # Save with lossless compression
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    cv2.imwrite(output_path, stego_image, compression_params)
    
    return output_path
    
    # Verify the saved image
    verification_image = cv2.imread(output_path, cv2.IMREAD_COLOR)
    if not np.array_equal(stego_image, verification_image):
        print("[WARNING] Compression has modified the image data")
    
    # Calculate actual bits stored
    stored_bits = bit_idx
    print(f"[INFO] Stego-image saved to {output_path}")
    print(f"[INFO] Successfully stored {stored_bits} bits out of {len(secret_bits)} bits")
    print(f"[INFO] Message integrity verified")
    print(f"[INFO] Used {bit_idx} bits out of {len(secret_bits)} bits")

if __name__ == "__main__":
    image_path = "input_image.png"
    output_path = "encoded_image.png"
    secret_text = ""

    embed_pvd(image_path, secret_text, output_path)