from mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore
import pickle
from tensorflow.keras.utils import image_dataset_from_directory #type: ignore
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type: ignore

# data = r"D:\Attendance Registration\fine_tune"

# full_ds = tf.keras.utils.image_dataset_from_directory(data, image_size=(128,128), batch_size=32)
# class_names = full_ds.class_names
# print("Classes:", class_names)

# with open("real_names.pkl", "wb") as f:
#     pickle.dump(class_names, f)

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data, validation_split=0.1, subset="training",
#     seed=42, image_size=(128,128), batch_size=32,
#     class_names=class_names)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data, validation_split=0.1, subset="validation",
#     seed=42, image_size=(128,128), batch_size=32,
#     class_names=class_names)

# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# model = tf.keras.models.load_model("best_cnn_model.h5")
# for layer in model.layers[:-1]:  # Freeze all layers except the last one
#     layer.trainable = True
# model.pop()  # remove last layer
# model.add(layers.Dense(3, activation="softmax", name="dense_3class"))

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.summary()
# es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# mc = ModelCheckpoint("fine_tuned_model.h5", save_best_only=True, monitor="val_accuracy")
# lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=20,
#     callbacks=[es, mc, lr])

with open("real_names.pkl", "rb") as f:
    class_names = pickle.load(f)

model = tf.keras.models.load_model("fine_tuned_model.h5")
detector = MTCNN()
def predict_face(img_path):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    x, y, w, h = faces[0]['box']   # take first face
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (128,128))
    face = np.expand_dims(face, axis=0)
    pred = model.predict(face) 
    idx = np.argmax(pred, axis=1)[0]
    print("Raw prediction vector:", pred[0])
    print("Argmax index:", np.argmax(pred[0]))
    print(f"Predicted: {class_names[idx]} ({100*np.max(pred[0]):.2f}% confidence)")

# predict_face(r"D:\Attendance Registration\lal.jpg")

cap = cv2.VideoCapture(0)
print("Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        predict_face(frame)   # pass frame directly
        break

cap.release()
cv2.destroyAllWindows()