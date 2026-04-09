import gradio as gr
import numpy as np
import pandas as pd
from tensorflow import keras

model = keras.models.load_model("student_model.h5")

df = pd.read_csv("student_performance_dataset_with_names.csv")

features = df.drop(columns=["name", "result"], errors="ignore").columns.tolist()

def predict(*inputs):
    try:
        data = np.array(inputs, dtype=float).reshape(1, len(features))
        prediction = model.predict(data)
        prob = float(prediction[0][0])

        if prob > 0.5:
            result = "PASS ✅"
            confidence = f"{prob * 100:.2f}%"
        else:
            result = "FAIL ❌"
            confidence = f"{(1 - prob) * 100:.2f}%"

        return f"{result}\nConfidence: {confidence}"

    except Exception as e:
        return f"Error: {str(e)}"

inputs = [gr.Number(label=feature, value=0) for feature in features]

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="🎓 Student Performance Prediction System",
    description="Enter student details to predict PASS or FAIL using a TensorFlow model."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)