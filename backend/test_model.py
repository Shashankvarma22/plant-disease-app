import numpy as np
from PIL import Image
import tensorflow as tf
import os

def test_model():
    print("🧪 Testing Plant Disease Model...")
    
    try:
        # Load model
        model = tf.keras.models.load_model('plant_disease_model.keras')
        print("✅ Model loaded successfully")
        print(f"Model output shape: {model.output_shape}")
        
        # Test 1: Random green image (simulating a leaf)
        print("\n--- Test 1: Green leaf simulation ---")
        test_image = np.ones((224, 224, 3)) * [0, 100, 0]  # Green image
        test_image = test_image.astype(np.uint8)
        test_image = np.expand_dims(test_image / 255.0, axis=0)
        
        predictions = model.predict(test_image, verbose=0)
        
        print("Predictions on green image:")
        max_confidence = np.max(predictions[0])
        min_confidence = np.min(predictions[0])
        avg_confidence = np.mean(predictions[0])
        
        print(f"Max confidence: {max_confidence:.4f}")
        print(f"Min confidence: {min_confidence:.4f}")
        print(f"Avg confidence: {avg_confidence:.4f}")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        class_names = sorted([
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
            'Tomato_healthy'
        ])
        
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_3_indices):
            class_name = class_names[idx] if idx < len(class_names) else f"Unknown_{idx}"
            print(f"  {i+1}. {class_name}: {predictions[0][idx]:.4f}")
        
        # Test 2: Random noise image
        print("\n--- Test 2: Random noise ---")
        noise_image = np.random.rand(224, 224, 3) * 255
        noise_image = noise_image.astype(np.uint8)
        noise_image = np.expand_dims(noise_image / 255.0, axis=0)
        
        noise_predictions = model.predict(noise_image, verbose=0)
        noise_max_confidence = np.max(noise_predictions[0])
        print(f"Max confidence on random noise: {noise_max_confidence:.4f}")
        
        # Analysis
        print("\n--- Analysis ---")
        if max_confidence < 0.5:
            print("❌ MODEL ISSUE: Very low confidence even on simulated leaf image")
            print("   The model may be under-trained or have architecture issues")
        elif max_confidence > 0.8:
            print("✅ Model seems to be working with good confidence")
        else:
            print("⚠️  Model has moderate confidence - may need improvement")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == '__main__':
    test_model()