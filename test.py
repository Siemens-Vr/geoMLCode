import requests
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

url = "http://127.0.0.1:5000/predict"
file_path = "E:/_peter/Geothermal data/Testing/Fire or smoke/6.jpg"  # Replace with your image path

def send_image_for_prediction(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Prepare the file for sending
        files = {
            'file0': ('image.jpg', open(file_path, 'rb'), 'image/jpeg')
        }

        # Send the request
        response = requests.post(url, files=files)
        
        # Check if the request was successful
        response.raise_for_status()

        # Parse and return the JSON response
        return response.json()

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
    except ValueError as e:
        logger.error(f"Error parsing response: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

# Send the image and get the prediction
result = send_image_for_prediction(file_path)

# Print the result
if result:
    logger.info("Prediction result:")
    logger.info(result)
else:
    logger.error("Failed to get prediction.")