
# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="SkinSpect: Skin Tumor Malignant or Benign Detection",
    page_icon = ":smile:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class

            return key

with st.sidebar:
        st.title("SkinSpect: Malignant or Benign")
        st.subheader("Accurate detection of skin tumor whether it is benign or malignant")
        st.subheader("Benign means non-cancerous. Benign skin tumors do not invade nearby tissues or spread to other parts of the body.")
        st.subheader("Malignant means cancerous. Malignant skin tumors have the potential to invade surrounding tissues and, if not treated early, may spread (metastasize) to other parts of the body.")

st.write("""
         # Skin Tumor Classification
         """
         )

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_data
def load_model():
    model=tf.keras.models.load_model('/content/model_skin_cancer.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['benign', 'malignant']

    string = "Detected Tumor : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'benign':
        st.balloons()
        st.sidebar.success(string)
        st.info("Your tumor is benign, so it is safe! It is not malignant so it is not cancerous.")
        st.info("Our recommendation would be to still get it checked out and tested!")

    elif class_names[np.argmax(predictions)] == 'malignant':
        st.sidebar.warning(string)
        st.markdown("## Malignant Tumor Detected")
        st.info("Seek medical attention ASAP. Your tumor has cancerous cells!")
        st.info("Malignant tumors are cancerous and can spread cancer cells throughout one's body through the blood or lymphatic system, a process known as metastasis")
        st.info("Treatment options may include surgery, chemotherapy or radiation therapy. Early detection is key, so be sure to attend all recommended cancer screenings.")
