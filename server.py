from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
try:
    model = load_model('E:/AI captioning/Script/improved_geothermal_model.keras') # change this to where you have downloaded your model
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

SAVE_DIR = 'saved_images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def preprocess_image(image_file, filename):
    try:
        # Open image from file
        image = Image.open(image_file)
        logger.info(f"Image format: {image.format}, size: {image.size}, mode: {image.mode}")
        
        # Save the image
        save_path = os.path.join(SAVE_DIR, filename)
        image.save(save_path)
        logger.info(f"Image saved to {save_path}")

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and convert to array
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        logger.info(f"Image array shape after preprocessing: {image_array.shape}")

        return np.expand_dims(image_array, axis=0) / 255.0
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

class_names = {0: "Data-cluster(fire/smoke)", 1: "Natural Geothermal Field", 2: "Geothermal station"}
class_explanations = {
    0: ["The image shows dense smoke or flame-like patterns indicating fire or smoke presence.",
        "The color gradients and shapes resemble those found in fire, with areas of high contrast."],
    1: ["Geothermal natural features are visible, including hot springs or fumaroles.",
        "The image contains natural rock formations typical of geothermal environments."],
    2: ["Industrial structures such as cooling towers are visible.",
        "The arrangement of geometric shapes suggests a geothermal power station setup."]
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file0' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.to_dict()
    if not files:
        return jsonify({"error": "No selected file"}), 400

    results = []
    for index, (key, file) in enumerate(files.items()):
        try:
            filename = f"image{index}.jpg"
            logger.info(f"Processing image {index}...")
            
            # Preprocess the image
            input_image = preprocess_image(file, filename)
            if input_image is None:
                raise ValueError("Failed to process received image")

            # Make predictions
            predictions = model.predict(input_image)
            prediction = np.argmax(predictions[0])
            confidence = float(predictions[0][prediction])
            class_name = class_names.get(prediction, "Unknown")
            explanations = class_explanations.get(prediction, ["No explanation available"])

            results.append({
                "filename": filename,
                "prediction": class_name,
                "confidence": confidence,
                "explanations": explanations
            })
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            results.append({"filename": f"image{index}.jpg", "error": str(e)})

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

