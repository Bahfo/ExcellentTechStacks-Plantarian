import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SPECIES_LABELS = {
    0: "Apple",
    1: "Blueberry",
    2: "Cherry (including sour)",
    3: "Corn (maize)",
    4: "Peach",
    5: "Pepper, bell",
    6: "Potato",
    7: "Raspberry",
    8: "Soybean",
    9: "Strawberry",
    10: "Tomato"
}

CLASS_LABELS = {
    0: "Apple: Apple scab",
    1: "Apple: Cedar apple rust",
    2: "Apple: Healthy",
    3: "Blueberry: Healthy",
    4: "Cherry: Healthy",
    5: "Corn: Cercospora leaf spot / Gray leaf spot",
    6: "Corn: Common rust",
    7: "Corn: Healthy",
    8: "Corn: Northern Leaf Blight",
    9: "Peach: Bacterial spot",
    10: "Peach: Healthy",
    11: "Pepper, bell: Bacterial spot",
    12: "Pepper, bell: Healthy",
    13: "Potato: Early blight",
    14: "Potato: Healthy",
    15: "Potato: Late blight",
    16: "Raspberry: Healthy",
    17: "Soybean: Healthy",
    18: "Squash: Powdery mildew",
    19: "Strawberry: Healthy",
    20: "Strawberry: Leaf scorch",
    21: "Tomato: Early blight",
    22: "Tomato: Healthy",
    23: "Tomato: Late blight",
    24: "Tomato: Leaf Mold",
    25: "Tomato: Septoria leaf spot",
    26: "Tomato: Tomato mosaic virus",
}

CLASS_TO_SPECIES = {
    0: "Apple",
    1: "Apple",
    2: "Apple",
    3: "Blueberry",
    4: "Cherry",
    5: "Corn",
    6: "Corn",
    7: "Corn",
    8: "Corn",
    9: "Peach",
    10: "Peach",
    11: "Pepper, bell",
    12: "Pepper, bell",
    13: "Potato",
    14: "Potato",
    15: "Potato",
    16: "Raspberry",
    17: "Soybean",
    18: "Squash",
    19: "Strawberry",
    20: "Strawberry",
    21: "Tomato",
    22: "Tomato",
    23: "Tomato",
    24: "Tomato",
    25: "Tomato",
    26: "Tomato"
}

def preprocess_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = preprocess_input(img.astype(np.float32))

    return np.expand_dims(img, axis=0)

def run_inference(image_path, model_path="plant_model.keras"):
    print(f"\n--- Loading Keras Model: {model_path} ---")

    if not os.path.exists(model_path):
        print(f"Error: Target model '{model_path}' not found in current directory.")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)

    print(f"--- Processing Input Image: {image_path} ---")

    input_tensor = preprocess_image(image_path)

    print("--- Executing Model Inference Pass ---")

    predictions = model.predict(input_tensor, verbose=0)

    pred_class_idx = int(np.argmax(predictions[0]))
    class_confidence = float(predictions[0][pred_class_idx]) * 100

    species_name = CLASS_TO_SPECIES.get(pred_class_idx, "Unknown")
    class_name = CLASS_LABELS.get(pred_class_idx, f"ID {pred_class_idx}")

    print("\n" + "=" * 40)
    print("         REALISTIC TEST RESULTS         ")
    print("=" * 40)
    print(f"Predicted Species : {species_name}")
    print(f"Predicted Class   : {class_name} (Confidence: {class_confidence:.2f}%)")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_test_image.jpg>")
        sys.exit(1)

    target_image = sys.argv[1]
    run_inference(target_image)