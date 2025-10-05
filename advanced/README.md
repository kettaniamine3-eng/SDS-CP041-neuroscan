# 🔴 Advanced Track

## Project Overview

**NeuroScan** is a deep learning project where participants build an image classification model to detect the presence of brain tumors using MRI scans.
The dataset contains two classes:

* **yes** → MRI image with brain tumor
* **no** → MRI image without brain tumor

This track focuses on a simple CNN pipeline, image preprocessing, and deployment through a Streamlit app.


## Objectives

### 1. Exploratory Data Analysis (EDA)

* Inspect class balance (number of "yes" vs. "no" images)
* Inspect images size and standardize if necessary (e.g., 128x128 or 224x224)
* Check pixel intensity distribution (to check if resizing is necessary)
* Normalize pixel values to [0,1]

### 2. Model Development

* Build a **Convolutional Neural Network (CNN)** using TensorFlow/Keras or PyTorch
* Architecture: Conv2D → Pooling → Dense layers → Output sigmoid/softmax
* Use **Binary Cross-Entropy loss** with accuracy as the main metric
* Apply **data augmentation** (rotation, zoom, flip) to prevent overfitting
* Train with validation split to monitor generalization

**Baseline models** (optional):

* Simple CNN (custom-built)
* Compare with pre-trained transfer learning model (e.g., MobileNetV2, ResNet50)


### 3. Model Deployment

- 🟢 Easy: **Streamlit Cloud**

  - Use the Streamlit app structure from the beginner track
  - Add deep learning model integration and host on Streamlit Community Cloud

- 🟡 Intermediate: **Docker + Hugging Face Spaces**

  - Containerize your Streamlit app using a custom `Dockerfile`
  - Deploy the Docker image to Hugging Face Spaces using the Docker SDK option

- 🔴 Advanced: **API-based Deployment (Flask or FastAPI)**

  - Create a RESTful API to serve model predictions
  - Containerize with Docker
  - Deploy to platforms like Railway, Render, Fly.io, or GCP Cloud Run
  - Validate using tools like Postman or a simple frontend client

## Technical Requirements

* **Data Handling & Visualization**: `numpy`, `pandas`, `matplotlib`, `seaborn`
* **Deep Learning**: `tensorflow` / `keras` (or PyTorch if preferred)
* **Image Processing**: `opencv`, `PIL`
* **Deployment**: `streamlit`


## Workflow & Timeline

| Phase                     | Core Tasks                                                | Duration      |
| ------------------------- | --------------------------------------------------------- | ------------- |
| **1 · Setup + EDA**       | Load dataset, visualize, preprocess, answer EDA questions | **Week 1**    |
| **2 · Model Development** | Train CNN, evaluate accuracy, tune hyperparameters        | **Weeks 2–3** |
| **3 · Deployment**        | Build Streamlit app, test locally, deploy to cloud        | **Week 4**    |

