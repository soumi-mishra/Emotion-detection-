Emotion Detection Project

Overview

This project is an end-to-end Emotion Detection Pipeline that integrates both facial and textual emotion recognition models into a single interactive Streamlit application. It leverages Convolutional Neural Networks (CNNs) for face emotion detection and LSTM-based models for text emotion detection, providing a unified interface for analyzing emotions from multiple modalities.

# Emotion Detection Project
[![Deploy on Render](https://img.shields.io/badge/Live%20App-Emotion%20Detection-blue?style=for-the-badge&logo=render)](https://emotion-detection-3-fxrp.onrender.com)

Features

Face Emotion Detection: Uses CNN models to classify emotions from facial images.

Text Emotion Detection: Employs LSTM models to analyze emotions from textual input.

Interactive Dashboard: Built with Streamlit, offering a clean and user-friendly interface.

Deployment Ready: Configured for deployment on Render with reproducible workflows.

Reproducible Environment: Python version pinned via pyproject.toml to ensure TensorFlow compatibility.

Tech Stack

Frontend: Streamlit

Backend Models: TensorFlow, Keras

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Image Processing: OpenCV (headless)

Installation

Clone the repository:

git clone https://github.com/soumi-mishra/Emotion-detection-
cd Emotion-detection-

Install dependencies:

pip install -r req.txt

Run the application locally:

streamlit run integration_app.py

Deployment

This project is deployed on Render. You can access the live application using the button below:



Project Goals

Provide a seamless interface for emotion detection from both text and facial inputs.

Ensure reproducible and clean deployment workflows.

Make data analysis projects interactive, accessible, and visually clear.

Author

Developed by Soumi Mishra

This README serves as a professional documentation of the Emotion Detection project, including setup, features, and deployment details.
