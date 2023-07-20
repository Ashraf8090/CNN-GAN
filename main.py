
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

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

# CNN for Reflection Mask Estimation
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN architecture
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    
    cnn = Model(inputs, x, name='CNN')
    return cnn

# Define the GAN
def build_gan(generator, cnn):
    input_shape = generator.input_shape[1:]  # Assuming the same input shape for both generator and cnn
    inputs = Input(shape=input_shape)
    
    # GAN architecture
    reflection_mask = cnn(inputs)
    reflection_free_image = generator([inputs, reflection_mask])
    
    gan = Model(inputs, reflection_free_image, name='GAN')
    return gan
# Loss functions
adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse_loss = tf.keras.losses.MeanSquaredError()
perceptual_loss = tf.keras.losses.MeanSquaredError()

lambda_adversarial = 1.0
lambda_mse = 100.0
lambda_perceptual = 10.0
lambda_tv = 1.0

# GAN loss (adversarial loss)
def gan_loss(y_true, y_pred):
    return lambda_adversarial * adversarial_loss(y_true, y_pred)

# Reflection mask loss (binary cross-entropy)
def mask_loss(y_true, y_pred):
    return lambda_mse * mse_loss(y_true, y_pred)

# Image content loss (perceptual loss)
def content_loss(y_true, y_pred):
    # Pre-trained VGG or ResNet model
    # Example: pre_trained_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    # Example: feature_true = pre_trained_model(y_true)
    # Example: feature_pred = pre_trained_model(y_pred)
    return lambda_perceptual * perceptual_loss(feature_true, feature_pred)

# Total Variation Loss
def tv_loss(y_pred):
    return lambda_tv * tf.image.total_variation(y_pred)

# Final loss function for the GAN
def generator_loss(y_true, y_pred):
    return gan_loss(y_true, y_pred) + mask_loss(y_true, y_pred) + content_loss(y_true, y_pred) + tv_loss(y_pred)

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

# Function to remove reflections using the trained GAN
def remove_reflections(images, generator):
    reflection_free_images = generator.predict(images)
    return reflection_free_images

# Define the input shape (assuming grayscale images of size 128x128)
input_shape = (128, 128, 1)

# Build the generator, CNN, and GAN
generator = build_generator(input_shape)
cnn = build_cnn(input_shape)
gan = build_gan(generator, cnn)

# Compile the GAN with the custom loss function
gan.compile(loss=generator_loss, optimizer='adam')

# Prepare data paths
training_dir = 'training'
model_dir = 'model'
output_dir = 'results_with_metrics'
os.makedirs(output_dir, exist_ok=True)

# Load training data
X_train = []
M_train = []
for filename in os.listdir(training_dir):
    if filename.endswith('.png'):
        image = tf.keras.preprocessing.image.load_img(os.path.join(training_dir, filename), target_size=input_shape[:2], color_mode='grayscale')
        reflection_mask = tf.keras.preprocessing.image.load_img(os.path.join(training_dir, filename.replace('.png', '_mask.png')), target_size=input_shape[:2], color_mode='grayscale')
        X_train.append(tf.keras.preprocessing.image.img_to_array(image))
        M_train.append(tf.keras.preprocessing.image.img_to_array(reflection_mask))
X_train = np.array(X_train) / 255.0
M_train = np.array(M_train) / 255.0

# Training loop
num_epochs = 10  # Adjust the number of epochs as needed
for epoch in range(num_epochs):
    # Generate fake reflections using the generator
    fake_reflections = generator.predict(X_train)
    
    # Train the discriminator (CNN) on real and fake reflection masks
    cnn.trainable = True
    cnn.train_on_batch(X_train, M_train)
    cnn.train_on_batch(fake_reflections, np.zeros_like(fake_reflections))
    
    # Train the generator (GAN) to fool the discriminator
    cnn.trainable = False
    gan.train_on_batch(X_train, M_train)
    
    # Print training progress
    print(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {loss}")

    # Save intermediate outputs with annotations
    for i, image in enumerate(fake_reflections):
        # Generate reflection-free image using the trained GAN
        reflection_free_image = generator.predict(np.expand_dims(X_train[i], axis=0))[0]

        # Calculate PSNR and SSIM between original reflection-free image and generated reflection-free image
        psnr_value = peak_signal_noise_ratio(X_train[i], reflection_free_image)
        ssim_value = structural_similarity(X_train[i], reflection_free_image)

        # Annotate the reflection-free image with PSNR and SSIM values
        annotated_image = annotate_image_with_metrics(Image.fromarray((reflection_free_image * 255).astype(np.uint8)), psnr_value, ssim_value)

        # Save the annotated image
        output_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{i+1}_psnr_{psnr_value:.2f}_ssim_{ssim_value:.4f}.png")
        annotated_image.save(output_path)

# Save the final model
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'reflection_removal_model.h5')
gan.save(model_path)

# Perform reflection removal on input images (e.g., in /input directory)
input_images_dir = 'input'
os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_images_dir):
    if filename.endswith('.png'):
        image = tf.keras.preprocessing.image.load_img(os.path.join(input_images_dir, filename), target_size=input_shape[:2], color_mode='grayscale')
        input_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        reflection_free_image = remove_reflections(np.expand_dims(input_image, axis=0), generator)[0]
        
        # Calculate PSNR and SSIM between original input image and reflection-free image
        psnr_value = peak_signal_noise_ratio(input_image, reflection_free_image)
        ssim_value = structural_similarity(input_image, reflection_free_image)

        # Annotate the reflection-free image with PSNR and SSIM values
        annotated_image = annotate_image_with_metrics(Image.fromarray((reflection_free_image * 255).astype(np.uint8)), psnr_value, ssim_value)
        
        # Save reflection-free image with annotations
        output_path = os.path.join(output_dir, f"reflection_free_{filename}")
        annotated_image.save(output_path)

print("Reflection removal completed and annotated output images saved.")
