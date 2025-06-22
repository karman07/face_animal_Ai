import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import random
import cloudinary
import cloudinary.uploader
from io import BytesIO
from datetime import datetime

# ✅ Set page config FIRST
st.set_page_config(page_title="What Animal Are You?", layout="centered")

# ====== Cloudinary Config ======
cloudinary.config(
    cloud_name="drp3gxgau",      # e.g., "karman-cloud"
    api_key="873195791534616",            # e.g., "1234567890"
    api_secret="FIpX1wBJ69wKUyndHcjhZx7hgMg",      # e.g., "abcdEfghIjklMNopqr"
    secure=True
)

# ====== Load Model ======
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_features_model_v2.h5")

model = load_model()

selected_features = ['Male', 'No_Beard', 'Smiling', 'Young', 'Eyeglasses', 'Big_Lips',
                     'Bald', 'Receding_Hairline', 'Wearing_Hat', 'Chubby', 'Double_Chin']

def get_animal_name(attributes):
    if attributes['Male'] == 1 and attributes['Beard'] == 1 and attributes['Smiling'] == 0:
        return "Lion"
    elif attributes['Smiling'] == 1 and attributes['Beard'] == 0 and attributes['Young'] == 1:
        return "Monkey"
    elif attributes['Eyeglasses'] == 1 and attributes['Chubby'] == 1:
        return "Panda"
    elif attributes['Eyeglasses'] == 1 or attributes['Big_Lips'] == 1:
        return "Owl"
    elif attributes['Bald'] == 1 or attributes['Receding_Hairline'] == 1:
        return "Snake"
    elif attributes['Wearing_Hat'] == 1:
        return "Duck"
    elif attributes['Chubby'] == 1 and attributes['Double_Chin'] == 1 and attributes['Smiling'] == 0:
        return "Bear"
    elif attributes['Chubby'] == 1 and attributes['Smiling'] == 1:
        return "Pig"
    elif attributes['Smiling'] == 1 and attributes['Young'] == 1 and attributes['Beard'] == 0 and attributes['Chubby'] == 0:
        return "Cat"
    elif attributes['Male'] == 1 and attributes['Beard'] == 1 and attributes['Smiling'] == 1:
        return "Tiger"
    elif attributes['Male'] == 1 and attributes['Beard'] == 0 and attributes['Smiling'] == 0 and attributes['Young'] == 1:
        return "Fox"
    else:
        return "Dog"

def get_random_animal_image(animal_class):
    folder = os.path.join(animal_class)
    if not os.path.isdir(folder):
        return None
    images = [f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not images:
        return None
    return os.path.join(folder, random.choice(images))

# ===== Streamlit UI =====
st.title("🐾 What Animal Do You Look Like?")
st.caption("Upload your photo and discover your inner animal!")

uploaded_file = st.file_uploader("📸 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))

    # Prediction
    img_arr = np.array(image) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_arr)[0]
    pred_binary = (pred > 0.5).astype(int)
    predicted_attrs = dict(zip(selected_features, pred_binary))
    predicted_attrs['Beard'] = 1 - predicted_attrs['No_Beard']

    animal_class = get_animal_name(predicted_attrs)
    emoji_map = {
        "Lion": "🦁", "Monkey": "🐵", "Panda": "🐼", "Owl": "🦉", "Snake": "🐍",
        "Duck": "🦆", "Bear": "🐻", "Pig": "🐷", "Cat": "🐱", "Tiger": "🐯", "Fox": "🦊", "Dog": "🐶"
    }
    result_emoji = emoji_map.get(animal_class, "🐾")

    # ===== Upload to Cloudinary =====
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    upload_result = cloudinary.uploader.upload(
        buffer,
        folder="animal_classifier",
        public_id=f"{animal_class}_{timestamp}",
        tags=[animal_class],
        resource_type="image"
    )
    uploaded_url = upload_result.get("secure_url")

    # ===== Result UI =====
    st.markdown("---")
    st.subheader(f"You look like a: {result_emoji} {animal_class}")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="👤 Your Photo", use_column_width=True)

    with col2:
        random_image_path = get_random_animal_image(animal_class)
        if random_image_path:
            animal_img = Image.open(random_image_path)
            st.image(animal_img, caption=f"{result_emoji} {animal_class}", use_column_width=True)
        else:
            st.warning(f"No example image found for {animal_class}.")


    with st.expander("🔍 Show Predicted Traits"):
        st.json(predicted_attrs)

# ===== 📄 About Section =====
st.markdown("---")
with st.expander("📄 About This App"):
    st.markdown("""
### 🐾 What Animal Do You Look Like?

This AI-powered app uses deep learning to analyze your facial features and match you to an animal!

### ☁️ Cloud Upload
Every uploaded image is saved to **Cloudinary** with the predicted animal label and current timestamp.

### 👨‍🎓 Made By
**Karman**, 3rd-year student at **Thapar Polytechnic College**.
""")
