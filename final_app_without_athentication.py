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
def load_model():
    """Load the fine-tuned model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path is now fixed to your best model
    model_path = "./models/blip-ft/final_blip_model3"
    
    if not os.path.isdir(model_path):
        st.error(f"Model not found at path: {model_path}")
        return None, None, None
        
    try:
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        st.write(f"Model loaded successfully on {device.upper()}.")
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

st.title("Image Description Generator 🖼️")
st.write("Upload an image and the AI will generate a description of that image. This model was fine-tuned on a subset (20,000 images) of the Flickr30k dataset.")

# Load the model and processor
model, processor, device = load_model()

if model:
    # --- Image Upload and Description Generation ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # This line is now fixed to use the new parameter
        st.image(image, caption="Your Uploaded Image", use_container_width=True)

        with st.spinner("Generating description..."):
            inputs = processor(images=image, return_tensors="pt").to(device)
            generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=100)
            generated_description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            st.success(f"**Generated Description:** {generated_description}")

