import gradio as gr
import numpy as np
import pandas as pd
from tensorflow import keras

model = keras.models.load_model("model.keras", compile=False)

df = pd.read_csv("student_performance_dataset_with_names.csv", encoding="utf-8")

features = df.drop(columns=["name", "result"], errors="ignore").columns.tolist()

def predict(*inputs):
    data = np.array(inputs, dtype=float).reshape(1, len(features))
    prediction = model.predict(data)
    prob = float(prediction[0][0])

    if prob > 0.5:
        return f"PASS ✅\nConfidence: {prob * 100:.2f}%"
    else:
        return f"FAIL ❌\nConfidence: {(1 - prob) * 100:.2f}%"

inputs = [gr.Number(label=feature, value=0) for feature in features]

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Student Performance Prediction System",
    description="Enter student details to predict PASS or FAIL"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)