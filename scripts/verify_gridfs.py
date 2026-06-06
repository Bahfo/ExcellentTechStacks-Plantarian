import os
import cv2
import numpy as np
from pymongo import MongoClient
import gridfs

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("DB_NAME", "plantarian_db")]

fs = gridfs.GridFS(db, collection="images_preprocessed")
col = db["metadata_preprocessed"]

doc = col.find_one()
img_bytes = fs.get(doc["gridfsFileId"]).read()

arr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

print("Shape:", img.shape)
print("Type:", img.dtype)