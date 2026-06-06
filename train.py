import os
import io
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, mixed_precision
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pymongo import MongoClient
import gridfs

mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("DB_NAME", "plantarian_db")]
fs = gridfs.GridFS(db, collection="images_preprocessed")
col = db["metadata_preprocessed"]

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

def augment_batch(image, label):
    return data_augmentation(image, training=True), label

def mongo_generator(split="train"):
    cursor = col.find({"datasetSplit": split})

    for doc in cursor:
        try:
            grid_out = fs.get(doc["gridfsFileId"])
            img_array = np.frombuffer(grid_out.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img.astype(np.float32))

            yield img, doc["classLabel"]

        except Exception:
            continue

def build_dataset(split, batch_size=16):
    dataset = tf.data.Dataset.from_generator(
        lambda: mongo_generator(split),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)

    if split == "train":
        dataset = dataset.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def build_model():
    base_model = applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(
        38,
        activation="softmax",
        dtype="float32",
        name="classLabel"
    )(x)

    model = models.Model(
        inputs=base_model.input,
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    print("Building Data Pipelines from MongoDB...")
    STEPS_PER_EPOCH = 2056 
    VALIDATION_STEPS = 440
    
    train_ds = build_dataset("train", batch_size=16)
    val_ds = build_dataset("validation", batch_size=16)
    
    model = build_model()

    lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', 
        factor=0.5, patience=2, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', 
        patience=4, restore_best_weights=True)
    
    checkpoint = callbacks.ModelCheckpoint(
        "plant_model_checkpoint.keras", 
        monitor="val_loss", 
        save_best_only=True,
        verbose=1
    )
    
    print("Starting High-Performance Python Training...")
    try:
        model.fit(
            train_ds,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=val_ds,
            validation_steps=VALIDATION_STEPS,
            epochs=15, 
            callbacks=[lr_decay, early_stop, checkpoint]
        )
    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user. Saving current weights safely...")
        model.save("plant_model_interrupted.keras")
        print("Interrupted weights saved to plant_model_interrupted.keras")
        import sys; sys.exit(0)
    
    model.save("plant_model.keras")
    print("Model saved to plant_model.keras")