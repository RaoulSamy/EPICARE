# EPICARE: Comprehensive Skin Cancer Diagnostic and Management System

## Description:
Epicare is a comprehensive platform designed to address the diagnosis and management of skin cancer, with a focus on Basal Cell Carcinoma (BCC) and melanoma. This platform integrates advanced machine learning algorithms for image classification, personalized treatment recommendations, and symptom tracking to provide a seamless experience for patients and healthcare providers alike. The system also incorporates an AI-powered chatbot that provides real-time assistance to patients, improving accessibility to information and ongoing support.

## Key Features:
- **Image Classification**: Classifies skin lesion images to detect BCC or melanoma using CNN-based models.
- **Personalized Recommendations**: Provides hospital and treatment recommendations based on patient data.
- **Symptom Tracking**: Allows patients to log and monitor their symptoms over time.
- **AI Chatbot**: Offers real-time, AI-driven responses to patient queries using a pre-defined dataset and advanced NLP models.
- **Progression Prediction**: Predicts the progression of skin cancer based on patient history and ongoing symptoms.

## Tech Stack:
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask, Python, MongoDB
- **Machine Learning Models**: TensorFlow/PyTorch, VGG16
- **Natural Language Processing**: Groq API, Meta LLaMA3-70B-8192
- **Database**: MongoDB (for chatbot queries and interactions), CSV datasets for models

## How to Use:
1. **Install dependencies**: Run `pip install -r requirements.txt`.
2. **Run the Flask app**: Run the command `python app.py`.
3. **Upload images**: Upload images for classification or interact with the chatbot through the web interface.
4. **Receive recommendations**: Receive recommendations and monitor symptoms via the dashboard.

## Project Highlights:
- **Accurate Skin Cancer Detection**: Utilizes state-of-the-art deep learning models to classify skin lesions and predict disease progression.
- **User-Centric Design**: Simplifies the interaction between patients and healthcare providers through an easy-to-use interface and personalized suggestions.
- **Real-time Assistance**: An AI-powered chatbot provides continuous support for patients during their treatment journey.
