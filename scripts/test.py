import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SPECIES_LABELS = {
    0: "Apple", 1: "Blueberry", 2: "Cherry (including sour)", 3: "Corn (maize)", 4: "Grape", 5: "Orange", 6: "Peach",
    7: "Pepper, bell", 8: "Potato", 9: "Raspberry", 10: "Soybean", 11: "Squash", 12: "Strawberry", 13: "Tomato"
}

CLASS_LABELS = {
    0: "Apple: Apple scab",
    1: "Apple: Black rot",
    2: "Apple: Cedar apple rust",
    3: "Apple: Healthy",
    4: "Blueberry: Healthy",
    5: "Cherry: Powdery mildew",
    6: "Cherry: Healthy",
    7: "Corn: Cercospora leaf spot / Gray leaf spot",
    8: "Corn: Common rust",
    9: "Corn: Northern Leaf Blight",
    10: "Corn: Healthy",
    11: "Grape: Black rot",
    12: "Grape: Esca (Black Measles)",
    13: "Grape: Leaf blight (Isariopsis Leaf Spot)",
    14: "Grape: Healthy",
    15: "Orange: Haunglongbing (Citrus greening)",
    16: "Peach: Bacterial spot",
    17: "Peach: Healthy",
    18: "Pepper, bell: Bacterial spot",
    19: "Pepper, bell: Healthy",
    20: "Potato: Early blight",
    21: "Potato: Late blight",
    22: "Potato: Healthy",
    23: "Raspberry: Healthy",
    24: "Soybean: Healthy",
    25: "Squash: Powdery mildew",
    26: "Strawberry: Leaf scorch",
    27: "Strawberry: Healthy",
    28: "Tomato: Bacterial spot",
    29: "Tomato: Early blight",
    30: "Tomato: Late blight",
    31: "Tomato: Leaf Mold",
    32: "Tomato: Septoria leaf spot",
    33: "Tomato: Spider mites (Two-spotted spider mite)",
    34: "Tomato: Target Spot",
    35: "Tomato: Tomato Yellow Leaf Curl Virus",
    36: "Tomato: Tomato mosaic virus",
    37: "Tomato: Healthy"
}

CLASS_TO_SPECIES = {
    0: "Apple", 1: "Apple", 2: "Apple", 3: "Apple", 4: "Blueberry", 5: "Cherry", 6: "Cherry", 7: "Corn", 8: "Corn", 9: "Corn", 10: "Corn",
    11: "Grape", 12: "Grape", 13: "Grape", 14: "Grape", 15: "Orange", 16: "Peach", 17: "Peach", 18: "Pepper, bell", 19: "Pepper, bell",
    20: "Potato", 21: "Potato", 22: "Potato", 23: "Raspberry", 24: "Soybean", 25: "Squash", 26: "Strawberry", 27: "Strawberry",
    28: "Tomato", 29: "Tomato", 30: "Tomato", 31: "Tomato", 32: "Tomato", 33: "Tomato", 34: "Tomato", 35: "Tomato", 36: "Tomato", 37: "Tomato"
}

def preprocess_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    # Match Python 'powerful_preprocess': Resize -> CLAHE (on L channel) -> Gaussian Blur
    img = cv2.resize(img, target_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    final_img = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
    
    # MobilenetV2 normalization [-1, 1]
    final_img = preprocess_input(final_img.astype(np.float32))

    return np.expand_dims(final_img, axis=0)

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