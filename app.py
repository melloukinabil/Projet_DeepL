import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.set_page_config(
    page_title='🐱🐶 Cats vs Dogs Classifier',
    page_icon='🐾',
    layout='centered'
)

@st.cache_resource
def load_model():
    return ort.InferenceSession('cats_dogs_model.onnx')

session = load_model()

# ------- UI -------
st.title('🐱🐶 Cats vs Dogs Classifier')
st.markdown(
    """
    Chargez une image de **chien** ou de **chat** et le modèle CNN (MobileNetV2 fine-tuné)
    vous indiquera sa prédiction avec un niveau de confiance.
    """
)

uploaded_file = st.file_uploader(
    'Choisissez une image (jpg / jpeg / png)...',
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Image chargée', width=350)

    # Prétraitement
    img_resized = img.resize((150, 150))
    img_array   = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

    # Prédiction
    with st.spinner('Analyse en cours...'):
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        proba = float(session.run([output_name], {input_name: img_array})[0][0][0])

    label = 'Chien 🐶' if proba > 0.5 else 'Chat 🐱'
    conf  = proba if proba > 0.5 else 1.0 - proba

    st.markdown(f'## Prédiction : **{label}**')
    st.markdown(f'**Confiance : {conf * 100:.1f}%**')
    st.progress(conf)

    with st.expander('Détails techniques'):
        st.write(f'Probabilité brute (sigmoid) : `{proba:.4f}`')
        st.write('`1 = Chien`, `0 = Chat`  (threshold = 0.5)')
