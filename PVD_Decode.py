import cv2
import numpy as np

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
    return range_table[-1]

def extract_pvd_channel(channel_data):
    """Extract bits from a single channel using PVD."""
    extracted_bits = ""
    flat_channel = channel_data.flatten()
    i = 0
    
    while i < len(flat_channel) - 1:
        p1 = flat_channel[i]
        p2 = flat_channel[i+1]
        
        diff = abs(int(p1) - int(p2))
        lower, upper, bits_to_extract = find_range(diff)
        
        # Extract the hidden value
        hidden_value = diff - lower
        new_bits = format(hidden_value, f'0{bits_to_extract}b')
        extracted_bits += new_bits
        
        i += 2
    
    return extracted_bits

def extract_pvd(stego_image_path):
    """Extract hidden message from stego image."""
    stego_image = cv2.imread(stego_image_path, cv2.IMREAD_COLOR)
    if stego_image is None:
        raise ValueError("Stego image not found or could not be loaded.")

    # Split into channels
    b, g, r = cv2.split(stego_image)
    
    # Extract from each channel sequentially
    extracted_bits = ""
    
    # Process all channels and collect all bits
    for channel in [r, g, b]:
        channel_bits = extract_pvd_channel(channel)
        extracted_bits += channel_bits
        
        # Check for end marker in complete bit string
        if '1111111111111110' in extracted_bits:
            marker_pos = extracted_bits.index('1111111111111110')
            extracted_bits = extracted_bits[:marker_pos]
            break
    
    # Convert bits back to text
    try:
        return bits_to_text(extracted_bits)
    except Exception as e:
        raise ValueError(f"Failed to extract message: {str(e)}")

if __name__ == "__main__":
    stego_image_path = "stego_image.png"
    try:
        extracted_message = extract_pvd(stego_image_path)
        print(f"[INFO] Extracted message: {extracted_message}")
    except ValueError as e:
        print(f"[ERROR] {str(e)}")