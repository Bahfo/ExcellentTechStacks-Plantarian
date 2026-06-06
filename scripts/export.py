import os
# Bypass the Protobuf version mismatch error safely
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"

import tensorflow as tf
import tensorflowjs as tfjs

MODEL_PATH = "plant_model.keras"
OUTPUT_DIR = "web_model"

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading target Keras model file...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # In Keras 3, save_format="tf" is replaced by model.export()
    temp_saved_model = "temp_tf_saved_model"
    print("Generating intermediate TensorFlow SavedModel directory via Keras 3 export...")
    model.export(temp_saved_model)

    # Use the converter module to compile optimized web graph shards
    print("Compiling optimized web graph models...")
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir=temp_saved_model,
        output_dir=OUTPUT_DIR
    )

    print(f"\n[SUCCESS] Saved production TensorFlow.js Graph model to: {OUTPUT_DIR}/model.json")

if __name__ == "__main__":
    main()