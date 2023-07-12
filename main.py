import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# GAN Generator
def build_generator(input_shape):
    inputs = Input(shape=input_shape)
    # Define the generator architecture using Convolutional and Upsampling layers
    # Example architecture:
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    upsampling1 = UpSampling2D((2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(upsampling1)
    conv4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv3)
    outputs = conv4
    model = Model(inputs=inputs, outputs=outputs)
    return model

# CNN for Reflection Mask Estimation
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    # Define the CNN architecture for reflection mask estimation
    # VGG Net (Architectural Defination):
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    pooling = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pooling)
    conv4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv3)
    outputs = conv4
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the GAN
def build_gan(generator, cnn):
    # Fix the weights of the CNN during GAN training
    cnn.trainable = False
    # Connect the generator and the CNN
    inputs = generator.input
    outputs = cnn(generator(inputs))
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the input shape (assuming grayscale images of size 128x128)
input_shape = (128, 128, 1)

# Build the generator, CNN, and GAN
generator = build_generator(input_shape)
cnn = build_cnn(input_shape)
gan = build_gan(generator, cnn)

# Compile the GAN
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Prepare data paths
training_dir = 'training'
model_dir = 'model'
input_dir = 'input'
results_dir = 'results'

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
    
    # Print training progress or save intermediate outputs

# Save the model
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'reflection_removal_model.h5')
gan.save(model_path)
print(f"Model saved at {model_path}")

# Load the model
gan = tf.keras.models.load_model(model_path)

# Inference on test images
os.makedirs(results_dir, exist_ok=True)
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        image = tf.keras.preprocessing.image.load_img(os.path.join(input_dir, filename), target_size=input_shape[:2], color_mode='grayscale')
        input_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        predicted_mask = gan.predict(input_image)
        predicted_mask = np.squeeze(predicted_mask, axis=0)
        predicted_mask *= 255.0
        predicted_mask = predicted_mask.astype(np.uint8)
        output_image = Image.fromarray(predicted_mask, mode='L')
        output_path = os.path.join(results_dir, f"{filename}_result.png")
        output_image.save(output_path)
        print(f"Result saved at {output_path}")
