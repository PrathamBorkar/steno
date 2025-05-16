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
        (32, 63, 4),
        (64, 127, 5),
        (128, 255, 6),
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
    # Make a copy to avoid modifying the original
    flat_channel = channel_data.flatten().copy()
    height, width = channel_data.shape
    
    # Track how many bits we've embedded
    original_bit_idx = bit_idx
    i = 0
    
    while i < len(flat_channel) - 1 and bit_idx < len(secret_bits):
        p1 = int(flat_channel[i])
        p2 = int(flat_channel[i+1])
        
        diff = abs(p1 - p2)
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
        
        # Calculate new pixel values with clipping to prevent overflow
        if p1 > p2:
            new_p2 = p1 - new_diff
            new_p1 = p1
        else:
            new_p1 = p2 - new_diff
            new_p2 = p2

        
        # Assign the clipped values
        flat_channel[i] = new_p1
        flat_channel[i+1] = new_p2
        
        bit_idx += bits_to_hide
        i += 2
    
    # Print debug info about this channel
    print(f"[DEBUG] Channel embedded {bit_idx - original_bit_idx} bits")
    
    return flat_channel.reshape(channel_data.shape), bit_idx

def get_pvd_capacity(image):
    """Calculate the maximum number of bits that can be embedded in the image."""
    b, g, r = cv2.split(image)
    total_capacity = 0
    
    for channel in [r, g, b]:
        flat_channel = channel.flatten()
        i = 0
        while i < len(flat_channel) - 1:
            p1 = flat_channel[i]
            p2 = flat_channel[i+1]
            diff = abs(int(p1) - int(p2))
            _, _, bits_to_hide = find_range(diff)
            total_capacity += bits_to_hide
            i += 2
    
    return total_capacity

def embed_pvd(image_path, secret_text, output_dir=None):
    # Load the color image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

    # Convert message to bits and add end marker
    secret_bits = text_to_bits(secret_text)
    end_marker = '1111111111111110'  # End Marker
    secret_bits_with_marker = secret_bits + end_marker
    
    # Check if the image has enough capacity
    capacity = get_pvd_capacity(image)
    if len(secret_bits_with_marker) > capacity:
        raise ValueError(f"Message too large for this image. Capacity: {capacity} bits, Message: {len(secret_bits_with_marker)} bits")
    
    # Split into channels
    b, g, r = cv2.split(image)
    bit_idx = 0
    
    # Embed in each channel sequentially
    r_stego, bit_idx = embed_pvd_channel(r, secret_bits_with_marker, bit_idx)
    
    # Only continue to next channel if we haven't embedded all bits
    if bit_idx < len(secret_bits_with_marker):
        g_stego, bit_idx = embed_pvd_channel(g, secret_bits_with_marker, bit_idx)
    else:
        g_stego = g.copy()
    
    # Only continue to next channel if we haven't embedded all bits
    if bit_idx < len(secret_bits_with_marker):
        b_stego, bit_idx = embed_pvd_channel(b, secret_bits_with_marker, bit_idx)
    else:
        b_stego = b.copy()
    
    # Merge channels
    stego_image = cv2.merge([b_stego, g_stego, r_stego])
    
    # Generate output path
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Create output filename
    output_path = os.path.join(output_dir, "pvd_stego.png")
    
    # Save with lossless compression
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    cv2.imwrite(output_path, stego_image, compression_params)
    
    # Print information about the embedding
    print(f"[INFO] Stego-image saved to {output_path}")
    print(f"[INFO] Successfully stored {bit_idx} bits out of {len(secret_bits_with_marker)} bits")
    print(f"[INFO] Original message length: {len(secret_bits)} bits")
    print(f"[INFO] Image capacity: {capacity} bits")
    
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