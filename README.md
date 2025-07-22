# Search Your Style(Snap Style)

This project implements a fashion recommender system using deep learning for feature extraction and a k-nearest neighbors algorithm for recommending similar images. It consists of three main scripts: one for feature extraction, one for a Streamlit web application, and one for running the recommender system from the command line.

## Technologies Used

- **TensorFlow**: Deep learning framework for loading a pre-trained ResNet50 model.
- **Keras**: High-level neural networks API for TensorFlow.
- **Streamlit**: Framework for creating interactive web applications.
- **Scikit-learn**: Machine learning library used for the k-nearest neighbors algorithm.
- **OpenCV**: Library for advanced image processing.
- **PIL (Python Imaging Library)**: Used for image loading and basic processing.
- **NumPy**: Library for numerical computations, essential for handling image arrays and feature vectors.
- **Pickle**: Python module for serializing and deserializing Python objects.

## Detailed Process and Working

### 1. Feature Extraction (initial script)

This script focuses on extracting features from images and saving them for future use in the recommendation system.

- **Model Setup**: 
  - Loads a pre-trained ResNet50 model without the top classification layer.
  - Adds a GlobalMaxPooling2D layer to reduce the dimensions of the feature maps.

- **Feature Extraction Function**:
  - Loads an image, resizes it, converts it to an array, and preprocesses it for compatibility with the ResNet50 model.
  - Predicts the features of the image, which are then flattened and normalized.

- **Feature Extraction Loop**:
  - Collects all image filenames from a specified directory.
  - Extracts features for each image using the defined function.

- **Saving Features and Filenames**:
  - Serializes and saves the extracted features and filenames using the pickle module.

### 2. Streamlit Application (`app.py`)

This script creates a web application that allows users to upload an image and get recommendations based on it.

- **Load Features and Filenames**:
  - Loads the previously saved features and filenames using pickle.

- **Model Setup**: 
  - Sets up the ResNet50 model with a GlobalMaxPooling2D layer, similar to the initial script.

- **Streamlit Setup**:
  - Initializes a Streamlit application and sets up the title.

- **File Upload and Save**:
  - Provides an interface for users to upload an image.
  - Saves the uploaded image locally and displays it on the web application.

- **Feature Extraction**: 
  - Processes the uploaded image and extracts its features using the same method as in the initial script.

- **Recommendation Function**:
  - Implements the k-nearest neighbors algorithm to find the most similar images based on the extracted features.

- **Display Recommendations**:
  - Uses Streamlit columns to display the recommended images.

### 3. Command Line Script (`main.py`)

This script enables running the recommendation system from the command line.

- **Load Features and Filenames**: 
  - Loads the saved features and filenames using pickle.

- **Model Setup**: 
  - Sets up the ResNet50 model similarly to the other scripts.

- **Feature Extraction**: 
  - Extracts features from a specific image (e.g., `sample/shirt.jpg`).

- **Nearest Neighbors Setup**: 
  - Fits the k-nearest neighbors model with the extracted features.

- **Display Recommendations**:
  - Uses OpenCV to display the recommended images on the screen.

## How to Run

1. Open the main.py file and in the terminal type:
```
streamlit run main.py
```
2. Open index.html file and run in your local host.
3. Go to are new feature home->fwd->snap style.
4. Upload an image through the web interface to get recommendations.
5. Explore the website: we also have cart option in the women section.

### Feature Extraction

1. Ensure you have a directory named `images` containing the images you want to process.
2. Run the feature extraction script to extract features and save them:
   ```bash
   python feature_extraction.py
   ```

### Streamlit Application

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open .html file on local host : index.html file

3. Upload an image through the web interface to get recommendations.

### Command Line Script

1. Run the command line script to get recommendations for a specific image:
   ```bash
   python main.py
   ```

## Directory Structure

```
.
├── css/                            # Directory containing CSS files
│   ├── index.css
│   ├── women.css
│   ├── bag.css
│   ├── fwd.css
│   └── search.css                  # For snap style feature 
├── scripts/                        # Directory containing JavaScript files
│   ├── bag.js
│   ├── item.js
│   ├── women.js
│   └── index.js
├── himage/                         # Directory containing all the website images
├── images/                         # Directory containing images to process
├── sample/                         # Directory containing sample images for command line script
│   └── shirt.jpg
├── uploads/                        # Directory where uploaded images are saved
├── embeddings.pkl                  # Serialized file containing extracted features
├── filenames.pkl                   # Serialized file containing image filenames
├── feature_extraction.py           # Script for extracting and saving image features
├── main.py                         # Command line script for recommendations
├── index.html                      # HTML file for the main page
├── fwd.html                        # HTML file for the FWD page
├── search.html                     # HTML file for the search page
├── women.html                      # HTML file for the women page
├── bag.html                        # HTML file for the bag page
├── app.py                          # Streamlit application script
└── README.md                       # This README file
```

## Dependencies

- TensorFlow
- Keras
- Streamlit
- Scikit-learn
- OpenCV
- PIL (Python Imaging Library)
- NumPy
- Pickle
- tqdm

Install the dependencies using:
```bash
pip install tensorflow keras streamlit scikit-learn opencv-python pillow numpy tqdm
```

---

This `README.md` file provides a comprehensive overview of the project, its components, and instructions on how to run it.
