from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
import os

# Convert any image format to PNG and return the new file path
def convert_to_png(image_path):
    """Convert any image format to PNG and return the new file path."""
    img = Image.open(image_path)
    png_path = os.path.splitext(image_path)[0] + ".png"
    img.save(png_path, "PNG")
    return png_path

# Convert string to binary bits with terminator
def string_to_bits(s):
    """Convert a string into binary bits with terminator."""
    # Add a special terminator pattern (8 bytes of zeros) to mark the end of the message
    s += '\0\0\0\0\0\0\0\0'
    return ''.join(format(ord(c), '08b') for c in s)

# Convert binary bits back to string, handling terminator
def bits_to_string(b):
    """Convert binary bits back into a string, stopping at the terminator."""
    result = ""
    for i in range(0, len(b), 8):
        if i + 8 <= len(b):
            byte = b[i:i+8]
            char_code = int(byte, 2)
            if char_code == 0:
                # Check for terminator sequence (8 consecutive zero bytes)
                if i + 64 <= len(b):
                    next_bytes = b[i:i+64]
                    if next_bytes == '0' * 64:
                        break
            result += chr(char_code)
    return result

# Embed a secret message into an image using DCT
def embed_message(image_path, message, output_path="watermarked_image.png"):
    """Embed a secret message into an image using DCT."""
    image_path = convert_to_png(image_path)

    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure image is in RGB format
    img_np = np.array(img)
    
    blue_channel = img_np[:, :, 2]  # Choose blue channel for embedding the message

    bits = string_to_bits(message)  # Convert message to binary with terminator
    bit_idx = 0
    block_size = 8  # Define block size for DCT
    h, w = blue_channel.shape  # Get the image dimensions

    stego_channel = np.copy(blue_channel)
    
    # Store the length of the message in the first 32 bits (4 bytes)
    msg_length = len(message)
    msg_length_bits = format(msg_length, '032b')
    
    # Embed message length
    for idx in range(32):
        i, j = idx // 4, (idx % 4) * 2
        block = blue_channel[i:i+block_size, j:j+block_size]
        
        if block.shape[0] != block_size or block.shape[1] != block_size:
            continue
            
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        coeff = dct_block[4, 4]
        int_coeff = int(round(coeff))
        bit_to_embed = int(msg_length_bits[idx])
        
        if (int_coeff % 2) != bit_to_embed:
            int_coeff += 1 if int_coeff % 2 == 0 else -1
            
        dct_block[4, 4] = np.float32(int_coeff)
        
        idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        stego_channel[i:i+block_size, j:j+block_size] = np.clip(idct_block, 0, 255)

    # Use a predefined starting point for the actual message embedding
    # This ensures synchronization during extraction
    start_i, start_j = 8, 0
    i, j = start_i, start_j

    # Loop over the image in blocks to embed the actual message
    while bit_idx < len(bits):
        if i >= h - block_size or j >= w - block_size:
            break
            
        block = blue_channel[i:i+block_size, j:j+block_size]

        if block.shape[0] != block_size or block.shape[1] != block_size:
            j += block_size
            if j >= w:
                j = 0
                i += block_size
            continue

        # Apply DCT to the block
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        coeff = dct_block[4, 4]  # Choose the coefficient for embedding the bit
        int_coeff = int(round(coeff))
        
        if bit_idx < len(bits):
            bit_to_embed = int(bits[bit_idx])  # Embed the bit

            # Modify the coefficient to store the bit
            if (int_coeff % 2) != bit_to_embed:
                int_coeff += 1 if int_coeff % 2 == 0 else -1

            dct_block[4, 4] = np.float32(int_coeff)

            # Apply Inverse DCT to the modified block
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            stego_channel[i:i+block_size, j:j+block_size] = np.clip(idct_block, 0, 255)

            bit_idx += 1

        # Move to the next block
        j += block_size
        if j >= w:
            j = 0
            i += block_size

    # Merge channels back to form the stego image
    stego_img_np = np.copy(img_np)
    stego_img_np[:, :, 2] = stego_channel  # Replace the blue channel with the stego one
    stego_img = Image.fromarray(stego_img_np)
    stego_img.save(output_path, "PNG")

    print(f"✅ Stego image saved as '{output_path}'.")
    return output_path

# Extract the hidden message from the image using DCT
def extract_message(image_path):
    """Extract a hidden message from a stego image."""
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure image is in RGB format
    img_np = np.array(img)
    
    blue_channel = img_np[:, :, 2]  # Use the blue channel for extraction

    block_size = 8  # Define block size for DCT
    h, w = blue_channel.shape
    
    # First, extract the message length from the header
    msg_length_bits = ""
    for idx in range(32):
        i, j = idx // 4, (idx % 4) * 2
        block = blue_channel[i:i+block_size, j:j+block_size]
        
        if block.shape[0] != block_size or block.shape[1] != block_size:
            continue
            
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        coeff = dct_block[4, 4]
        int_coeff = int(round(coeff))
        
        msg_length_bits += str(int_coeff & 1)
        
    try:
        msg_length = int(msg_length_bits, 2)
        print(f"Detected message length: {msg_length} characters")
    except ValueError:
        print("Failed to extract message length")
        return "ERROR: Extraction failed!"
    
    # Extract with a safety factor to ensure we get the terminator
    total_bits = (msg_length + 20) * 8  
    extracted_bits = ""
    
    # Use the same predefined starting point as in embedding
    start_i, start_j = 8, 0
    i, j = start_i, start_j
    bits_read = 0
    
    # Loop over the image in blocks
    while bits_read < total_bits:
        if i >= h - block_size or j >= w - block_size:
            break
            
        block = blue_channel[i:i+block_size, j:j+block_size]

        if block.shape[0] != block_size or block.shape[1] != block_size:
            j += block_size
            if j >= w:
                j = 0
                i += block_size
            continue

        # Apply DCT to the block
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        coeff = dct_block[4, 4]  # Extract the coefficient for the bit
        int_coeff = int(round(coeff))

        extracted_bits += str(int_coeff & 1)  # Extract the least significant bit
        bits_read += 1

        # Move to the next block
        j += block_size
        if j >= w:
            j = 0
            i += block_size

    extracted_message = bits_to_string(extracted_bits)  # Convert extracted bits back to a string
    print(f"✅ Extracted message: {extracted_message}")
    return extracted_message

# Wrapper for embed_message function
def encode(image_path, message, output_dir=None):
    import os
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "stego_image.png")
    else:
        output_path = "stego_image.png"
    return embed_message(image_path, message, output_path)

# Wrapper for extract_message function
def decode(image_path):
    return extract_message(image_path)

# Example of usage
if __name__ == "__main__":
    image_path = 'input_image.png'
    message = 'Hello, this is a secret message!'
    encoded_image = encode(image_path, message)  # Embed the message
    decoded_message = decode(encoded_image)  # Extract the message
    print(f"Original message: {message}")
    print(f"Decoded message: {decoded_message}")
