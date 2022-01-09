import streamlit as st
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
MODEL_PATH ='model_vgg.h5'
model = load_model(MODEL_PATH)


st.title('Malaria-Detection')

img_path = st.text_input('Specify the path to cell image')

try:

    st.image(img_path, caption='Cell Image')


    img = image.load_img(img_path, target_size=(224, 224))
        # Preprocessing the image
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

        # Scaling
    x = x/255

    pred = model.predict(x)
    prediction = np.argmax(pred[0])
    if prediction == 0:
        p = "REPORT: The cell is Parasitized.       *Malaria Detected. "

    else:
        p = "REPORT: The cell is not Infected."

    st.subheader(p)





    #####
    ### Heatmap ###
    #####

    if prediction == 0:

        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()

        import tensorflow.keras.backend as K

        with tf.GradientTape() as tape:
            last_conv_layer = model.get_layer('block5_pool')
            iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(x)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
        
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)


        import matplotlib.pyplot as plt

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((7, 7))

        import cv2
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(img_path)

        INTENSITY = 0.5
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_HOT)
        img = heatmap * INTENSITY + img

        st.image(image=img, clamp=True, caption = 'Infected Area')

except:
    pass








hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            footer:after {
	            content:'Made by Nilavo Boral'; 
	            visibility: visible;
	            display: block;
	            position: relative;
	            #background-color: red;
	            padding: 5px;
	            top: 2px;
                color: tomato;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)