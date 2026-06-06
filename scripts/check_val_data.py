from train import build_dataset
import numpy as np

val_ds = build_dataset("validation", batch_size=16)

# Pull exactly one batch from the validation generator
for x, y in val_ds.take(1):
    print("--- VALIDATION DATA INTEGRITY CHECK ---")
    print("Image batch shape:", x.shape)
    
    class_labels = y.numpy()
    
    print("Class Labels in Batch:", class_labels)
    print("Min/Max Class Label:", np.min(class_labels), "/", np.max(class_labels))