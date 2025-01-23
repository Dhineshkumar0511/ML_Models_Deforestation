import openai
import numpy as np
from PIL import Image
import streamlit as st
import logging

# Set up logging to print errors to the terminal
logging.basicConfig(level=logging.DEBUG)

# OpenAI API function for analysis using the correct method for the latest version
def analyze_prediction_with_llm(deforestation_percentage):
    try:
        openai.api_key = "your api key"  # Replace with your OpenAI API key

        # Construct the prompt
        prompt = (
            f"The detected deforestation percentage from the satellite image is {deforestation_percentage:.2f}%. "
            "Provide an analysis of the environmental impact and possible mitigation strategies for such a level of deforestation."
        )

        # New API call method for openai v1.0.0 and above
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can replace this with other models like GPT-4
            messages=[
                {"role": "system", "content": "You are an assistant providing environmental analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        # Log the error in the terminal
        logging.error(f"Error occurred while calling OpenAI API: {e}")
        return "An error occurred while processing the request. Please try again later."

# Preprocess the satellite image
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Main function for the Streamlit app
def main():
    st.title("Deforestation Prediction from Satellite Image")

    # Upload image
    uploaded_image = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open the image
        image = Image.open(uploaded_image)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Simulate model prediction (since the model part isn't provided in the code)
        # In a real case, this would be replaced with your model's prediction
        # For now, we will assume a random deforestation percentage for demonstration
        deforestation_percentage = np.random.uniform(0, 100)  # Replace this with the model prediction

        st.write(f"Predicted Deforestation Percentage: {deforestation_percentage:.2f}%")

        # Analyze the prediction using OpenAI's API (error will be captured and logged)
        analysis = analyze_prediction_with_llm(deforestation_percentage)
        

if __name__ == "__main__":
    main()
