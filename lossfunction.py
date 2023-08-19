import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from PIL import Image, ImageDraw, ImageFont

# GAN Generator
def build_generator(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    skip_conn = x

    # Downsampling
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    skip_conn2 = x

    # Downsampling
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Upsampling
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, skip_conn2], axis=-1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Upsampling
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, skip_conn], axis=-1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Output
    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    generator = Model(inputs, output, name='Generator')
    return generator

# ... Rest of the code ...

# Calculate PSNR
def psnr(y_true, y_pred):
    max_value = 1.0  # Assuming pixel values are in the range [0, 1]
    return tf.image.psnr(y_true, y_pred, max_val=max_value)

# Calculate SSIM
def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

# Function to annotate image with PSNR and SSIM values
def annotate_image_with_metrics(image, psnr_value, ssim_value):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # You can use a custom font if needed
    draw.text((10, 10), f"PSNR: {psnr_value:.2f}", font=font, fill='white')
    draw.text((10, 30), f"SSIM: {ssim_value:.4f}", font=font, fill='white')
    return image

# ... Rest of the code ...

# Perform reflection removal on input images (e.g., in /input directory)
input_images_dir = 'input'
os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_images_dir):
    if filename.endswith('.png'):
        image = tf.keras.preprocessing.image.load_img(os.path.join(input_images_dir, filename), target_size=input_shape[:2], color_mode='grayscale')
        input_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        reflection_free_image = remove_reflections(np.expand_dims(input_image, axis=0), generator)[0]

        # Calculate PSNR and SSIM between original input image and reflection-free image
        psnr_value = psnr(np.expand_dims(input_image, axis=0), reflection_free_image)
        ssim_value = ssim(np.expand_dims(input_image, axis=0), reflection_free_image)

        # Annotate the reflection-free image with PSNR and SSIM values
        annotated_image = annotate_image_with_metrics(Image.fromarray((reflection_free_image * 255).astype(np.uint8)), psnr_value, ssim_value)

        # Save reflection-free image with annotations
        output_path = os.path.join(output_dir, f"reflection_free_{filename}")
        annotated_image.save(output_path)

print("Reflection removal completed and annotated output images saved.")
