from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

try:
    from model import PlantDiseaseModel
    # Initialize model
    model = PlantDiseaseModel('plant_disease_model.keras')
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# Disease information and treatment advice
DISEASE_INFO = {
    'Pepper__bell___Bacterial_spot': {
        'name': 'Pepper Bacterial Spot',
        'description': 'Bacterial disease causing small, water-soaked spots on leaves and fruits that turn brown and cracked.',
        'treatment': 'Use copper-based bactericides, practice crop rotation, and avoid overhead watering.'
    },
    'Pepper__bell___healthy': {
        'name': 'Healthy Bell Pepper',
        'description': 'The pepper plant appears to be healthy with no signs of disease.',
        'treatment': 'Continue with regular care, proper watering, and monitoring.'
    },
    'Potato___Early_blight': {
        'name': 'Potato Early Blight',
        'description': 'Fungal disease causing target-like concentric rings on leaves, stems, and tubers.',
        'treatment': 'Apply fungicides containing chlorothalonil, practice crop rotation, and remove infected plant debris.'
    },
    'Potato___Late_blight': {
        'name': 'Potato Late Blight',
        'description': 'Serious fungal disease that can destroy entire crops, causing water-soaked lesions that turn brown.',
        'treatment': 'Use fungicides like mancozeb, destroy infected plants, and ensure proper plant spacing.'
    },
    'Potato___healthy': {
        'name': 'Healthy Potato',
        'description': 'The potato plant appears to be healthy with no signs of disease.',
        'treatment': 'Continue with regular care, proper watering, and monitoring.'
    },
    'Tomato_Bacterial_spot': {
        'name': 'Tomato Bacterial Spot',
        'description': 'Bacterial disease causing small, dark, scabby spots on leaves, stems, and fruits.',
        'treatment': 'Use copper sprays, practice crop rotation, and avoid working with wet plants.'
    },
    'Tomato_Early_blight': {
        'name': 'Tomato Early Blight',
        'description': 'Fungal disease causing dark concentric rings on lower leaves, stems, and fruits.',
        'treatment': 'Apply fungicides, remove infected leaves, and ensure good air circulation.'
    },
    'Tomato_Late_blight': {
        'name': 'Tomato Late Blight',
        'description': 'Destructive fungal disease causing large, water-soaked lesions on leaves and fruits.',
        'treatment': 'Use appropriate fungicides, remove and destroy infected plants immediately.'
    },
    'Tomato_Leaf_Mold': {
        'name': 'Tomato Leaf Mold',
        'description': 'Fungal disease causing yellow spots on upper leaf surfaces with moldy growth underneath.',
        'treatment': 'Improve air circulation, use fungicides, and avoid overhead watering.'
    },
    'Tomato_Septoria_leaf_spot': {
        'name': 'Tomato Septoria Leaf Spot',
        'description': 'Fungal disease causing small, circular spots with dark borders and light centers on leaves.',
        'treatment': 'Remove infected leaves, apply fungicides, and practice crop rotation.'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'name': 'Tomato Spider Mites',
        'description': 'Tiny pests that suck plant juices, causing stippling, yellowing, and webbing on leaves.',
        'treatment': 'Use miticides, increase humidity, and remove heavily infested leaves.'
    },
    'Tomato__Target_Spot': {
        'name': 'Tomato Target Spot',
        'description': 'Fungal disease causing circular spots with concentric rings resembling targets.',
        'treatment': 'Apply fungicides, improve air circulation, and remove infected plant material.'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'description': 'Viral disease causing upward curling of leaves, yellowing, and stunted growth.',
        'treatment': 'Control whiteflies (vectors), remove infected plants, and use resistant varieties.'
    },
    'Tomato__Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus',
        'description': 'Viral disease causing mottled light and dark green patterns on leaves.',
        'treatment': 'Use virus-free seeds, control aphids, and practice good sanitation.'
    },
    'Tomato_healthy': {
        'name': 'Healthy Tomato',
        'description': 'The tomato plant appears to be healthy with no signs of disease.',
        'treatment': 'Continue with regular care, proper watering, and monitoring.'
    }
}

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        # Make prediction
        result = model.predict(file)
        
        # Add disease information
        disease_key = result['class']
        disease_info = DISEASE_INFO.get(disease_key, {
            'name': disease_key.replace('__', ' ').replace('___', ' ').replace('_', ' '),
            'description': 'No specific information available for this condition.',
            'treatment': 'Consult with agricultural experts for specific treatment recommendations.'
        })
        
        result.update({
            'disease_name': disease_info['name'],
            'description': disease_info['description'],
            'treatment': disease_info['treatment']
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    status = 'healthy' if model is not None else 'model not loaded'
    return jsonify({
        'status': status, 
        'message': 'Plant Disease Classification API',
        'model_loaded': model is not None,
        'total_classes': len(model.class_names) if model else 0
    })

if __name__ == '__main__':
    print("🚀 Starting Plant Disease Classification API...")
    print("📍 Key improvements in this version:")
    print("   ✅ Fixed mixed precision compatibility")
    print("   ✅ Improved image preprocessing (matches Keras defaults)")
    print("   ✅ Better error handling and logging")
    print("   ✅ Confidence-based warnings")
    print("🌐 Server: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)