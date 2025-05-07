from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from lsb import encode as lsb_encode, decode as lsb_decode
from dct import encode as dct_encode, decode as dct_decode
from patchwork import embed_watermark, extract_watermark
from PVD_Encode import embed_pvd
from PVD_Decode import extract_pvd
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def save_uploaded_file(file_storage):
    filename = secure_filename(file_storage.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(filepath)
    return filepath

import logging

# Add new imports at the top
from flask import send_from_directory
import os.path

# Add new route to serve images
@app.route('/get_image/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/encode', methods=['POST'])
def encode():
    if 'image' not in request.files:
        return jsonify(success=False, error='No image file provided')
    image_file = request.files['image']
    message = request.form.get('message', '')
    method = request.form.get('method', '').lower()
    save_dir = request.form.get('save_dir', '').strip()

    if not message and method != 'patchwork':
        return jsonify(success=False, error='No message provided for encoding')

    image_path = save_uploaded_file(image_file)

    # Determine output directory
    output_dir = app.config['UPLOAD_FOLDER']
    if save_dir:
        # Validate and create directory if needed
        if not os.path.isabs(save_dir):
            # Make relative paths relative to current working directory
            save_dir = os.path.abspath(save_dir)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as e:
                return jsonify(success=False, error=f"Failed to create directory: {str(e)}")
        output_dir = save_dir

    try:
        if method == 'lsb':
            output_path = lsb_encode(image_path, message, output_dir=output_dir)
        elif method == 'dct':
            output_path = dct_encode(image_path, message, output_dir=output_dir)
        elif method == 'pvd':
            output_path = embed_pvd(image_path, message, output_dir=output_dir)
        elif method == 'patchwork':
            output_path = embed_watermark(image_path, message, output_dir=output_dir)
        else:
            return jsonify(success=False, error='Invalid method')
    except Exception as e:
        logging.exception("Encoding failed")
        return jsonify(success=False, error=str(e))

    return jsonify(success=True, 
                  output_path=output_path,
                  output_filename=os.path.basename(output_path))

@app.route('/decode', methods=['POST'])
def decode():
    method = request.form.get('method', '').lower()

    if 'image' not in request.files:
        return jsonify(success=False, error='No image file provided')
    image_file = request.files['image']
    image_path = save_uploaded_file(image_file)

    try:
        if method == 'lsb':
            message = lsb_decode(image_path)
        elif method == 'dct':
            message = dct_decode(image_path)
        elif method == 'pvd':
            message = extract_pvd(image_path)
        elif method == 'patchwork':
            message = extract_watermark(image_path)
        else:
            return jsonify(success=False, error='Invalid method')
    except Exception as e:
        return jsonify(success=False, error=str(e))

    return jsonify(success=True, message=message)

from flask import send_file
import os

@app.route('/')
def serve_index():
    return send_file(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'index.html'))

if __name__ == '__main__':
    app.run(debug=True)
