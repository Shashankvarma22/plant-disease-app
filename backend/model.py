import numpy as np
from PIL import Image
import io
import logging
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantDiseaseModel:
    def __init__(self, model_path):
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('float32')

            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            logger.info("✅ Model loaded successfully!")
            logger.info(f"Model output shape: {self.model.output_shape}")

            # This class_names order MUST match your training folder directory list sorted alphabetically
            self.class_names = [
                'Pepper__bell___Bacterial_spot',
                'Pepper__bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Tomato_Bacterial_spot',
                'Tomato_Early_blight',
                'Tomato_Late_blight',
                'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot',
                'Tomato__Tomato_YellowLeaf__Curl_Virus',
                'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy'
            ]

            expected_classes = self.model.output_shape[-1]
            if len(self.class_names) != expected_classes:
                logger.error(f"❌ Class count mismatch: Model expects {expected_classes}, but we have {len(self.class_names)} classes")
                raise ValueError("Class count mismatch.")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def preprocess_image(self, image):
        try:
            # Convert image to RGB and resize to 224x224 (NO center crop)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            return image_array
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise

    def predict(self, image_file):
        try:
            image = Image.open(io.BytesIO(image_file.read()))
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)

            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]

            top_3_indices = np.argsort(predictions[0])[-3:][::-1]

            warning = ""
            if confidence < 0.3:
                warning = "❌ VERY LOW CONFIDENCE - Model is highly uncertain"
            elif confidence < 0.6:
                warning = "⚠️ LOW CONFIDENCE - Model is uncertain about this prediction"
            elif confidence < 0.8:
                warning = "ℹ️ MODERATE CONFIDENCE - Prediction should be verified"
            else:
                warning = "✅ HIGH CONFIDENCE - Prediction is reliable"

            all_predictions = {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}

            logger.info(f"Predicted class: {predicted_class} with confidence {confidence:.4f}")
            logger.info("Top 3 predictions:")
            for i, idx in enumerate(top_3_indices):
                logger.info(f"  {i+1}. {self.class_names[idx]}: {predictions[0][idx]:.4f}")

            return {
                'class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'warning': warning,
                'is_reliable': confidence > 0.6
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


app = Flask(__name__)
CORS(app)

try:
    model = PlantDiseaseModel('plant_disease_model.keras')
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400

    try:
        result = model.predict(file)
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
    app.run(debug=True, host='0.0.0.0', port=5000)
