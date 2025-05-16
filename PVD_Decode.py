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
    return range_table[-1]

def extract_pvd_channel(channel_data):
    """Extract bits from a single channel using PVD."""
    extracted_bits = ""
    flat_channel = channel_data.flatten().copy()  # Make a copy to avoid modifying original
    i = 0
    
    while i < len(flat_channel) - 1:
        p1 = int(flat_channel[i])
        p2 = int(flat_channel[i+1])
        
        diff = abs(p1 - p2)
        lower, upper, bits_to_extract = find_range(diff)
        
        # Extract the hidden value (subtract lower bound to get actual value)
        hidden_value = diff - lower
        
        # Format with correct number of bits
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
    end_marker = '1111111111111110'
    found_marker = False
    
    # Process channels in the same order as embedding
    for channel_idx, channel in enumerate([r, g, b]):
        channel_bits = extract_pvd_channel(channel)
        print(f"[DEBUG] Extracted {len(channel_bits)} bits from channel {channel_idx}")
        
        # Add to total extracted bits
        extracted_bits += channel_bits
        
        # Check for end marker
        if end_marker in extracted_bits:
            marker_pos = extracted_bits.index(end_marker)
            extracted_bits = extracted_bits[:marker_pos]
            found_marker = True
            print(f"[INFO] Found end marker after {marker_pos} bits")
            break
    
    if not found_marker:
        print("[WARNING] End marker not found. Message might be corrupted.")
    
    # Convert bits back to text
    try:
        message = bits_to_text(extracted_bits)
        return message
    except Exception as e:
        # If conversion fails, try to recover as much as possible
        print(f"[ERROR] Failed to convert all bits to text: {str(e)}")
        # Try to convert as many complete bytes as possible
        complete_bytes = len(extracted_bits) - (len(extracted_bits) % 8)
        if complete_bytes > 0:
            partial_bits = extracted_bits[:complete_bytes]
            return bits_to_text(partial_bits)
        else:
            raise ValueError("No valid message could be extracted")

if __name__ == "__main__":
    stego_image_path = "stego_image.png"
    try:
        extracted_message = extract_pvd(stego_image_path)
        print(f"[INFO] Extracted message: {extracted_message}")
    except ValueError as e:
        print(f"[ERROR] {str(e)}")