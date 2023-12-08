#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('intel_image.h5')

#Name of Classes
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

#Setting Title of App
st.title("Intel Image Classification")
st.markdown("Upload an image")

#Uploading the dog image
intel_image = st.file_uploader("Choose an image...", type=['jpg','png'])
submit = st.button('Predict')
#On predict button click
if submit:


    if intel_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(intel_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (150,150))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,150,150,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("The image is "+CLASS_NAMES[np.argmax(Y_pred)]))
