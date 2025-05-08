import cv2
import numpy as np
from scipy.fftpack import dct, idct
import os

# Standard JPEG luminance quantization matrix
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

def string_to_bits(text):
    """Convert a string into binary bits with terminator."""
    # Add a special terminator pattern to mark the end of the message
    text += '\0\0\0\0'
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_string(bits):
    """Convert binary bits back into a string, stopping at the terminator."""
    result = ""
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = bits[i:i+8]
            char_code = int(byte, 2)
            if char_code == 0:
                # Check for terminator sequence
                if i + 32 <= len(bits):
                    next_bytes = bits[i:i+32]
                    if next_bytes == '0' * 32:
                        break
            result += chr(char_code)
    return result

def embed_message(image_path, message, output_path="stego_image.png"):
    """Embed a secret message into an image using DCT with quantization."""
    # Load and resize image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or invalid format: {image_path}")
    
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = y.astype(np.float32)

    # Message to binary with magic header
    bin_msg = string_to_bits(message)
    
    # Store the message length first (16 bits)
    magic_header = "1010101010101010"  # Easy to recognize pattern
    msg_length_bin = format(len(message), '016b')  # 16 bits for length
    
    # Complete message: magic header + length + actual message
    bin_msg = magic_header + msg_length_bin + bin_msg
    total_bits = len(bin_msg)
    msg_index = 0

    h, w = y.shape
    for row in range(0, h-8, 8):
        for col in range(0, w-8, 8):
            if msg_index >= total_bits:
                break

            block = y[row:row+8, col:col+8]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quant_block = np.round(dct_block / Q)

            # Modify DC coefficient (top-left corner)
            dc = int(quant_block[0, 0])
            sign = 1 if dc >= 0 else -1
            
            # Embed bit in LSB of DC coefficient
            if msg_index < total_bits:
                dc_bin = format(abs(dc), '08b')
                new_dc = int(dc_bin[:-1] + bin_msg[msg_index], 2) * sign
                quant_block[0, 0] = new_dc
                msg_index += 1

            # Dequantize and inverse DCT
            dct_block = quant_block * Q
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            y[row:row+8, col:col+8] = idct_block

    # Reconstruct image
    y = np.clip(y, 0, 255).astype(np.uint8)
    stego_img = cv2.merge([y, cr, cb])
    stego_img = cv2.cvtColor(stego_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, stego_img)
    
    print(f"✅ Stego image saved as '{output_path}'.")
    return output_path

def extract_message(image_path):
    """Extract a hidden message from a stego image."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read")
    
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(ycrcb)
    y = y.astype(np.float32)

    h, w = y.shape
    
    # Extract all bits from DC coefficients
    all_bits = []
    for row in range(0, h-8, 8):
        for col in range(0, w-8, 8):
            block = y[row:row+8, col:col+8]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quant_block = np.round(dct_block / Q)
            dc = int(quant_block[0, 0])
            # Extract LSB regardless of sign
            all_bits.append(format(abs(dc), '08b')[-1])
    
    # Convert bits to string
    extracted_bits = ''.join(all_bits)
    
    # Look for the magic header
    magic_header = "1010101010101010"
    if magic_header not in extracted_bits[:100]:
        print("Magic header not found! Possibly corrupt data.")
        return "ERROR: Extraction failed!"
    
    # Find where our real data starts
    header_pos = extracted_bits.find(magic_header)
    data_start = header_pos + 16  # Skip magic header
    
    # Extract the length
    length_bits = extracted_bits[data_start:data_start + 16]
    try:
        msg_length = int(length_bits, 2)
        print(f"Detected message length: {msg_length} characters")
    except ValueError:
        print("Failed to extract message length")
        return "ERROR: Extraction failed!"
    
    # Extract the message
    message_start = data_start + 16  # Skip length bits
    message_bits = extracted_bits[message_start:message_start + (msg_length * 8) + 32]  # Add extra for terminator
    
    # Convert to text
    extracted_message = bits_to_string(message_bits)
    print(f"✅ Extracted message: {extracted_message}")
    return extracted_message

# Wrapper for embed_message function
def encode(image_path, message, output_dir=None):
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "dct_stego.png")
    else:
        output_path = "dct_stego.png"
    return embed_message(image_path, message, output_path)

# Wrapper for extract_message function
def decode(image_path):
    return extract_message(image_path)

# Example of usage
if __name__ == "__main__":
    image_path = 'input_image.png'
    message = 'Hello, this is a secret message!'
    encoded_image = encode(image_path, message)
    decoded_message = decode(encoded_image)
    print(f"Original message: {message}")
    print(f"Decoded message: {decoded_message}")
