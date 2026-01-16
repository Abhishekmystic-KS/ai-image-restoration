# AI Image Restoration with ESRGAN

This repository contains a Google Colab notebook demonstrating AI-powered image restoration using the ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) model from TensorFlow Hub. The provided code focuses on restoring low-resolution or degraded images, with optimizations for memory usage.

## Table of Contents

- Project Overview
- Features
- Setup
- Usage
- Model Details
- Example Output

## Project Overview

This project leverages a pre-trained ESRGAN model to enhance the quality of images. It's particularly useful for upscaling and improving the visual fidelity of older or lower-quality images. The notebook is designed to be run in Google Colab, making it accessible without requiring local GPU setup.

## Features

- **ESRGAN Model Integration**: Utilizes the captain-pool/esrgan-tf2/1 model from TensorFlow Hub.
- **RAM Optimization**: Automatically resizes large input images (e.g., above 600px) to prevent out-of-memory errors, especially when running on Colab or systems with limited RAM/GPU.
- **Color Correction**: Correctly handles pixel value scaling (0-255 range) to avoid common color issues like purple/magenta tints in the output.
- **Simple Python Interface**: A straightforward restore_image function to process images.

## Setup

To run this project, you will need a Google Colab environment.

1. **Open the Notebook**: Upload or open the .ipynb file in Google Colab.

2. **Install Dependencies**: Run the first code cell to install the necessary Python libraries:

```bash
pip install tensorflow tensorflow-hub numpy Pillow
```

3. **Import Libraries**: Ensure all required libraries are imported:

```python
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import os
```

## Usage

1. **Upload Your Image**: Upload the image you wish to restore to your Colab environment. For example, if your image is named my_image.jpg, upload it to the /content/ directory.

2. **Load the Model and Define Function**: The notebook contains a cell that loads the ESRGAN model and defines the restore_image function. This function handles loading, resizing, processing, and saving the image.

```python
# 1. Load the model once
print("Loading ESRGAN model...")
model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

def restore_image(image_path, output_path):
    # Load Image
    img = Image.open(image_path).convert('RGB')

    # --- RAM OPTIMIZATION: Resize if the image is too big ---
    # ESRGAN 4x will crash if the input is > 1000px.
    # We downsize the input so the output stays under control.
    max_size = 600
    if max(img.size) > max_size:
        print(f"Resizing input from {img.size} to save RAM...")
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Convert to Tensor (Removed / 255.0 normalization as model expects 0-255 range)
    img_tensor = tf.expand_dims(np.array(img).astype(np.float32), axis=0)

    print("Processing... (This uses high RAM/GPU)")
    # Run the model
    restored_tensor = model(img_tensor)

    # Convert back to PIL (Removed * 255.0 and added clipping for 0-255 range)
    restored_tensor = tf.squeeze(restored_tensor, axis=0)
    restored_img_array = tf.cast(tf.clip_by_value(restored_tensor, 0, 255), tf.uint8).numpy()
    restored_image = Image.fromarray(restored_img_array)

    # Save result
    restored_image.save(output_path)
    print(f"Done! Restored image saved as: {output_path}")
```

3. **Run the Restoration**: Call the restore_image function with your input image path and desired output path.

```python
restore_image('my_image.jpg', 'restored_my_image.png')
```

Replace 'my_image.jpg' with the path to your input image.

4. **Download the Result**: After execution, the restored image (e.g., restored_my_image.png) will be saved in your Colab environment, which you can then download.

## 🏗️ General Project Structure

The code is organized into a modular structure to ensure it is easy to read and memory-efficient:

1. **Environment Setup:** Checking for TensorFlow and ensuring the Streamlit server is active.
2. **Resource Loading:** Downloading the pre-trained ESRGAN model from TensorFlow Hub once and caching it.
3. **Image Pipeline:**
   - **Input:** Accepting user uploads via Streamlit.
   - **Preprocessing:** Resizing images to a safe range (< 600px) and converting them into mathematical tensors.
   - **Inference:** The AI model processes the image to add detail and sharpness.
   - **Post-processing:** Clipping values to remove "purple tint" artifacts and converting back to a viewable image format.
4. **Display/Export:** Showing the "Before and After" comparison and providing a download link.

## Example Output
 You can use markdown image links like ![Original Image](path/to/original.jpg) and ![Restored Image](path/to/restored.png))

## If have GPU 
 use V/S code
 use main_vs.py & requirements.txt
## Dont have GPU
  use CoLab
  use mainColab.ipynb
  
