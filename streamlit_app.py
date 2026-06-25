import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

model = load_model()

# =========================
# Preprocessing Image
# =========================
def preprocess_image(foto):
    # Resize sesuai input MobileNetV2
    foto = foto.resize((224, 224))

    # Pastikan RGB
    if foto.mode != "RGB":
        foto = foto.convert("RGB")

    # Convert ke numpy array
    image_array = np.array(foto)

    # Preprocessing MobileNetV2
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Tambah dimensi batch
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image

# =========================
# Tampilan Streamlit
# =========================
st.set_page_config(
    page_title="Klasifikasi Gambar MobileNetV2",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Klasifikasi Gambar dengan MobileNetV2")
st.write("Upload gambar JPG, JPEG, atau PNG untuk melakukan prediksi.")

# Upload file
uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Baca gambar
    foto = Image.open(uploaded_file)

    # Tampilkan gambar
    st.image(
        foto,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    if st.button("Prediksi"):

        with st.spinner("Sedang memproses gambar..."):

            # Preprocessing
            processed_image = preprocess_image(foto)

            # Prediksi
            predictions = model.predict(processed_image)

            # Decode hasil
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions,
                top=3
            )[0]

        st.success("Prediksi selesai!")

        st.subheader("Hasil Prediksi")

        for i, (_, label, prob) in enumerate(decoded_predictions):
            st.write(
                f"**{i+1}. {label.replace('_', ' ').title()}** : {prob*100:.2f}%"
            )

            st.progress(float(prob))

        # Prediksi terbaik
        best_label = decoded_predictions[0][1]
        best_prob = decoded_predictions[0][2]

        st.info(
            f"Prediksi utama: **{best_label.replace('_',' ').title()}** "
            f"({best_prob*100:.2f}%)"
        )
