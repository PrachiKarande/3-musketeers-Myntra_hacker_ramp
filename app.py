import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load ResNet50 model pre-trained on ImageNet without the top classification layer
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add a GlobalMaxPooling2D layer to the model
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# List to store image filenames
filenames = []

# Directory containing images
image_dir = 'images'

# Append image file paths to the filenames list
for file in os.listdir(image_dir):
    filenames.append(os.path.join(image_dir, file))

# List to store extracted features
feature_list = []

# Extract features for each image and append to the feature_list
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save the extracted features and filenames to pickle files
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

# Load the pickled features and filenames
with open('embeddings.pkl', 'rb') as f:
    feature_list = pickle.load(f)

with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

