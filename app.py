import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import cv2

model = models.load_model('face_emotion_classification.keras')

emotions = [['angry'], ['disgust'], ['fear'], ['happy'], ['neutral'], ['sad'], ['surprise']]

st.title('Face Emotion Classification')

image_path = st.text_input('Enter Image Path')

if st.button('Predict'):
  image = cv2.imread(image_path)[:, :, 0]
  image = cv2.resize(image, (48, 48))
  image = np.invert(np.array([image]))
  output = model.predict(image)
  np.argmax(output)
  output = emotions[np.argmax(output)]

  stn = 'Emotion in the image is ' + str(output[0])
  st.header(stn)
