from PIL import Image

import os

def encode(image_path, message, output_dir=None):
    import os
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "lsb_stego.png")
    else:
        output_path = "static/uploads/stego_image.png"

    img = Image.open(image_path)
    img = img.convert('RGB')
    binary_message = ''.join(format(ord(c), '08b') for c in message) + '1111111111111110'  # Delimiter

    data_index = 0
    img_data = img.getdata()
    new_data = []

    for pixel in img_data:
        if data_index < len(binary_message):
            r = pixel[0] & ~1 | int(binary_message[data_index])
            data_index += 1
        else:
            r = pixel[0]
        if data_index < len(binary_message):
            g = pixel[1] & ~1 | int(binary_message[data_index])
            data_index += 1
        else:
            g = pixel[1]
        if data_index < len(binary_message):
            b = pixel[2] & ~1 | int(binary_message[data_index])
            data_index += 1
        else:
            b = pixel[2]
        new_data.append((r, g, b))

    img.putdata(new_data)
    img.save(output_path, "PNG")
    return output_path

def decode(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_data = img.getdata()

    binary_message = ''
    for pixel in img_data:
        for color in pixel[:3]:
            binary_message += str(color & 1)

    # Split by 8 bits
    chars = [binary_message[i:i+8] for i in range(0, len(binary_message), 8)]
    message = ''
    for char in chars:
        if char == '11111111':  # Delimiter part 1
            next_char_index = chars.index(char) + 1
            if next_char_index < len(chars) and chars[next_char_index] == '11111110':  # Delimiter part 2
                break
        message += chr(int(char, 2))
    return message
