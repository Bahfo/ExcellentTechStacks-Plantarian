import os
import json
import cv2
import gridfs
import random
from pymongo import MongoClient
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("DB_NAME", "plantarian_db")]
fs_new = gridfs.GridFS(db, collection="images_preprocessed")
col_new = db["metadata_preprocessed"]

def powerful_preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    final_img = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()

def process_and_upload():
    print("Starting Preprocessing & MongoDB Upload...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    staging_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "staging_data"))

    col_new.drop()
    
    metadata_path = os.path.join(staging_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print("Performing global structural dataset randomization...")
    random.seed(42)
    random.shuffle(metadata)

    total_items = len(metadata)
    split_boundary = int(total_items * 0.85)

    for i, item in enumerate(metadata):
        img_path = os.path.join(staging_dir, item["filename"])
        
        current_split = "train" if i < split_boundary else "validation"
        
        try:
            processed_bytes = powerful_preprocess(img_path)
            file_id = fs_new.put(processed_bytes, filename=item["filename"])

            col_new.insert_one({
                "original_filename": item["filename"],
                "gridfsFileId": file_id,
                "classLabel": item["classLabel"],
                "speciesLabel": item["speciesLabel"],
                "datasetSplit": current_split
            })
        except Exception as e:
            print(f"\nError processing {item['filename']}: {e}. Skipping...")
            continue
        
        if i % 1000 == 0:
            print(f"Preprocessed & Uploaded: {i} / {total_items} (Assigned as: {current_split})")

    print("Preprocessing Complete!")

if __name__ == "__main__":
    process_and_upload()