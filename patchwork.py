import cv2
import numpy as np
import os

def text_to_bits(text):
    """Convert text to binary string"""
    return ''.join(format(ord(char), '08b') for char in text)

def bits_to_text(bits):
    """Convert binary string back to text"""
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

def embed_watermark(image_path, message, output_dir=None):
    """Embeds a message using the Patchwork Algorithm."""
    # Convert message to binary
    message_bits = text_to_bits(message)
    message_length = len(message_bits)
    
    # Read and prepare image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or invalid format: {image_path}")
    
    height, width = image.shape
    watermarked_image = image.astype(np.float64)
    
    # Parameters
    alpha = 3  # Reduced for better imperceptibility
    bits_per_block = 2  # Number of bits embedded per block
    block_size = 8  # Size of each block
    
    # Calculate required blocks
    blocks_needed = (message_length + bits_per_block - 1) // bits_per_block
    if blocks_needed * block_size * block_size > height * width:
        raise ValueError("Message too large for image")
    
    # Embed length first
    length_bits = format(message_length, '032b')
    
    # Embed message
    bit_index = 0
    for block_idx in range(blocks_needed + 4):  # +4 blocks for length
        row = (block_idx * block_size) % (height - block_size)
        col = ((block_idx * block_size) // (height - block_size)) * block_size
        
        if col >= width - block_size:
            raise ValueError("Image too small for message")
        
        block = watermarked_image[row:row+block_size, col:col+block_size]
        
        # Get current bits to embed
        if block_idx < 4:  # First 4 blocks for length
            current_bits = length_bits[block_idx * bits_per_block:(block_idx + 1) * bits_per_block]
        elif bit_index < message_length:
            current_bits = message_bits[bit_index:min(bit_index + bits_per_block, message_length)]
            bit_index += bits_per_block
        
        # Modify block based on bits
        if len(current_bits) > 0:
            for bit_idx, bit in enumerate(current_bits):
                if bit == '1':
                    block[bit_idx:bit_idx+4, :] += alpha
                    block[bit_idx+4:bit_idx+8, :] -= alpha
    
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    
    # Generate output path
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_patchwork.png")
    
    cv2.imwrite(output_path, watermarked_image)
    return output_path

def extract_watermark(image_path):
    """Extracts the hidden message from a watermarked image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be read")
    
    height, width = image.shape
    block_size = 8
    bits_per_block = 2
    
    # Extract length first
    length_bits = ""
    for block_idx in range(4):
        row = (block_idx * block_size) % (height - block_size)
        col = ((block_idx * block_size) // (height - block_size)) * block_size
        block = image[row:row+block_size, col:col+block_size]
        
        for bit_idx in range(bits_per_block):
            upper_half = np.mean(block[bit_idx:bit_idx+4, :])
            lower_half = np.mean(block[bit_idx+4:bit_idx+8, :])
            length_bits += '1' if upper_half > lower_half else '0'
    
    message_length = int(length_bits, 2)
    blocks_needed = (message_length + bits_per_block - 1) // bits_per_block
    
    # Extract message
    message_bits = ""
    for block_idx in range(4, blocks_needed + 4):
        row = (block_idx * block_size) % (height - block_size)
        col = ((block_idx * block_size) // (height - block_size)) * block_size
        block = image[row:row+block_size, col:col+block_size]
        
        for bit_idx in range(bits_per_block):
            if len(message_bits) < message_length:
                upper_half = np.mean(block[bit_idx:bit_idx+4, :])
                lower_half = np.mean(block[bit_idx+4:bit_idx+8, :])
                message_bits += '1' if upper_half > lower_half else '0'
    
    try:
        return bits_to_text(message_bits[:message_length])
    except:
        raise ValueError("Failed to extract valid message")
