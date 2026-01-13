import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf 
import numpy as np
from PIL import Image

#loadnthe ESRGiAN model (cache it so it loas only once)
@st.cache_resource
def load_model():
    model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')
    return model

# Convert PIL image to TensorFlow tensor
def pil_to_tensor(pil_image):
    # Convert PIL image to numpy array
    img_array = np.array(pil_image)
    
    # Convert to float32 and normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension [1, H, W, 3]
    img_tensor = tf.expand_dims(img_array, axis=0)
    
    return img_tensor

# Convert tensor back to PIL image
def tensor_to_pil(tensor):
    # Remove batch dimension
    tensor = tf.squeeze(tensor, axis=0)
    
    # Convert to uint8 [0, 255]
    img_array = tf.cast(tensor * 255.0, tf.uint8).numpy()
    
    # Convert to PIL image
    pil_img = Image.fromarray(img_array)
    
    return pil_img

# Streamlit app
def main():
    st.title("🖼️ AI Image Restoration App")
    st.write("Upload an old/damaged image → AI restores it!")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file).convert('RGB')
        
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)
        
        # Process button
        if st.button("✨ Restore Image"):
            st.write("Processing... this may take 30-60 seconds...")
            
            # Load model and convert image
            model = load_model()
            img_tensor = pil_to_tensor(original_image)
            
            # Run AI restoration
            restored_tensor = model(img_tensor)
            restored_image = tensor_to_pil(restored_tensor)
            
            # Display restored image
            st.subheader("Restored Image")
            st.image(restored_image, use_column_width=True)
            
            # Download button
            restored_image.save("restored_image.png")
            with open("restored_image.png", "rb") as file:
                st.download_button(
                    label="📥 Download Restored Image",
                    data=file.read(),
                    file_name="restored_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
