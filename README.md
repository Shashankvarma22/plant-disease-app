🌿 Plant Disease Classification and Detection Using Deep Learning

A full-stack machine learning application that detects plant leaf diseases using image classification and provides actionable insights for crop management.

🚀 Overview

Plant diseases significantly impact agricultural productivity. Manual identification is often time-consuming and requires expert knowledge.

This project presents a deep learning-based plant disease detection system that automates disease classification from leaf images using a trained convolutional neural network and integrates it into a web application for real-time usage.

🧠 Problem Statement

Early detection of plant diseases is critical for reducing crop loss and improving yield. Traditional methods are:

Time-consuming
Dependent on expert knowledge
Not scalable for large or remote farms

This system provides an automated, scalable solution for disease identification.

🎯 Objectives
Develop a deep learning model for plant disease classification
Use the PlantVillage dataset (~20,000+ images, 15 classes)
Apply preprocessing and augmentation for better generalization
Integrate the model into a Flask backend API
Build a React frontend for user interaction
🏗️ System Architecture
User (React Frontend)
        ↓
Flask Backend API
        ↓
TensorFlow/Keras Model (MobileNetV2)
        ↓
Prediction (Disease + Confidence)
        ↓
Frontend Display

As described in your report, the system follows a 3-layer architecture: frontend, backend, and ML model .

🧪 Model Details
Architecture: MobileNetV2 (Transfer Learning)
Input Size: 224 × 224 × 3
Dataset: PlantVillage
Classes: 15 (Pepper, Potato, Tomato diseases + healthy)
Training Strategy:
Pretrained on ImageNet
Fine-tuned on plant dataset
Data augmentation (flip, rotation, zoom)
Accuracy: ~98.7%
⚙️ Tech Stack
Backend
Python
Flask
TensorFlow / Keras
NumPy, Pillow
Frontend
React.js
Axios
🔌 API Endpoints
POST /api/predict

Upload an image and get prediction.

Request:

multipart/form-data
file: image

Response:

{
  "prediction": "Tomato Early Blight",
  "confidence": 0.92,
  "description": "...",
  "treatment": "...",
  "top_predictions": [...]
}
GET /api/health
{
  "status": "healthy",
  "model_loaded": true
}
📦 Installation & Setup
1. Clone Repository
git clone https://github.com/Shashankvarma22/plant-disease-app.git
cd plant-disease-app
2. Backend Setup
cd backend
python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
python app.py

Backend runs on: http://localhost:5000

3. Frontend Setup
cd frontend
npm install
npm start

Frontend runs on: http://localhost:3000

📊 Features
🌱 Image-based plant disease classification
📈 Confidence score with predictions
💊 Treatment recommendations
🧠 Deep learning-based inference
⚡ Real-time web interface
🌍 Applications
Farmer support systems
Agricultural advisory tools
Research and crop monitoring systems

As highlighted in your report, the system can assist farmers by providing instant disease predictions without expert dependency .

⚠️ Limitations
Limited to 15 disease classes
Trained on controlled dataset
May struggle with real-world variations
🔮 Future Scope
Real-time detection using camera
Mobile app integration
Expansion to more crops and diseases
Edge deployment (TensorFlow Lite)
🧾 Conclusion

This project demonstrates how deep learning + web technologies can solve real-world agricultural problems. The MobileNetV2-based model provides accurate and efficient predictions, making it suitable for practical deployment in smart farming systems .