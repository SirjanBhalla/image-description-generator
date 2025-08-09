import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Description Generator",
    page_icon="📸",
    layout="centered"
)

# --- Model Loading (with Caching) ---
@st.cache_resource
def load_model(model_name):
    """Load a specified fine-tuned model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Construct the path to the selected model folder
    model_path = os.path.join("./models/blip-ft", model_name)
    
    if not os.path.isdir(model_path):
        st.error(f"Model '{model_name}' not found at path: {model_path}")
        return None, None, None
        
    try:
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        st.write(f"Model '{model_name}' loaded on {device.upper()}.")
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        return None, None, None

st.title("Image Description Generator 🖼️")
st.write("Upload an image and select a model to generate a caption.")

# --- Model Selection Dropdown ---
# Path where your model folders are located
models_dir = "./models/blip-ft"

if os.path.isdir(models_dir):
    # Get a list of available models from the directory
    available_models = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]
else:
    available_models = []

if not available_models:
    st.error(f"No models found in the '{models_dir}' directory. Please make sure your model folders are there.")
else:
    selected_model = st.selectbox("Choose a model to test:", available_models)

    # Load the selected model
    model, processor, device = load_model(selected_model)

    if model:
        # --- Image Upload and Description Generation ---
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your Uploaded Image", use_column_width=True)
            
            with st.spinner("Generating a one line description..."):
                inputs = processor(images=image, return_tensors="pt").to(device)
                generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
                generated_description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                st.success(f"**Generated Description:** {generated_description}")
