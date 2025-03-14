# brain_tumor_model.py

import numpy as np
import matplotlib.pyplot as plt
import os
import gradio as gr
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Data Preparation ---

# Define dataset directories
base_dir = os.path.join(os.getcwd(), "data")
train_dir = os.path.join(base_dir, "training")
validation_dir = os.path.join(base_dir, "validation")

# Function to Train Model & Return Plots
def train_model(epochs, batch_size):
    """Trains the model based on user input (epochs & batch size). Returns accuracy and loss plots."""
    
    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=batch_size, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(64, 64), batch_size=batch_size, class_mode='binary')

    # --- Model Building ---
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # --- Model Training ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[early_stopping]
    )

    # Save model
    model.save("brain_tumor_model.keras")

    # --- Generate Plots ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax[0].plot(history.history["accuracy"], label="Training Accuracy")
    ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    # Loss plot
    ax[1].plot(history.history["loss"], label="Training Loss")
    ax[1].plot(history.history["val_loss"], label="Validation Loss")
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    return fig  # Return performance graph

# --- Prediction Function ---

def predict_uploaded_image(image_array):
    """Processes an uploaded MRI scan, converts it to RGB, resizes it, and predicts if it contains a tumor."""
    model = load_model("brain_tumor_model.keras")  # Load trained model

    try:
        # Convert NumPy array to a PIL image
        img = Image.fromarray(image_array.astype("uint8"))

        # Convert to RGB (if grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to model input size (64x64)
        img = img.resize((64, 64))

        # Convert image to array & normalize
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)[0][0]

        # Return formatted prediction result
        if prediction > 0.5:
            return f"🧠 Tumor Detected! (Confidence: {prediction * 100:.2f}%)"
        else:
            return f"✅ No Tumor Detected. (Confidence: {(1 - prediction) * 100:.2f}%)"

    except Exception as e:
        return f"⚠️ Error processing image: {str(e)}"

# --- Gradio Interactive UI ---

with gr.Blocks() as interface:
    gr.Markdown("# 🧠 Brain Tumor Detection AI")
    
    with gr.Tab("Train Model"):
        gr.Markdown("### Set training parameters:")
        epochs = gr.Number(value=10, label="Epochs")
        batch_size = gr.Number(value=32, label="Batch Size")
        train_button = gr.Button("Train Model")
        train_output = gr.Plot()  # Display training plots
        train_button.click(train_model, inputs=[epochs, batch_size], outputs=train_output)

    with gr.Tab("Upload MRI Scan"):
        gr.Markdown("### Upload an MRI scan for prediction:")
        img_input = gr.Image(type="numpy")
        pred_output = gr.Textbox(label="Prediction Result")
        predict_button = gr.Button("Analyze MRI")
        predict_button.click(predict_uploaded_image, inputs=img_input, outputs=pred_output)

# Launch Gradio interface
interface.launch()
