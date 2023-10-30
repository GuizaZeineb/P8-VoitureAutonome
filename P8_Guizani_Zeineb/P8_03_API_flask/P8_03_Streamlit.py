



import streamlit as st
import json, requests


import pandas as pd
import numpy as np
import os


from PIL import Image

import matplotlib.pyplot as plt

from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt


import string



def convert_reception(img):
    #Obtenir l'image
    img = base64.b64decode(img) #Convertir les données d'image converties en base64 en données binaires d'origine# bytes
    img = BytesIO(img) # _io.Converti pour être géré par l'oreiller BytesIO
    return img
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:pink;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Image Segmentation using flask</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    #url1 = "http://127.0.0.1:5000/"
    url1 = "https://p8-flask-app.azurewebsites.net/"
    


    request1 = requests.get(url=url1)
    response1 = request1.text
    st.write(response1)

    #2- listing the images
    st.header("List of images")
    #url1 = "http://127.0.0.1:5000/api/get_img_list/"
    url1 = "https://p8-flask-app.azurewebsites.net/api/get_img_list"
    request1 = requests.get(url=url1)
    response1 = request1.json()
    id_image = st.selectbox("Select Image Id", [image for image in response1["data"]])
    st.write(id_image)    


    
    
        
    #3- Récupération de l'image et du masque
    #url2 = "http://127.0.0.1:5000/api/get_images"
    url2 = "https://p8-flask-app.azurewebsites.net/api/get_images"


    response = {'data': id_image}
    
    request2 = requests.post(url=url2, json=response)
    data = request2.json()
    image = data["img"]
    image = convert_reception(image)

    
    mask = data["mask"]
    mask = convert_reception(mask)
    #st.write("Preview image")    
    #st.image(Image.open(image), width=None)



    #3- Prediction
 
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict Mask"): 

        #url3 = "http://127.0.0.1:5000/api/prediction"
        url3 = "https://p8-flask-app.azurewebsites.net/api/prediction"


        #headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-cache'}
    
        response = {'data': id_image}
        #json.dumps(response)
        
        request3 = requests.post(url=url3, json=response)
        data = request3.json()
        predicted_mask = data["img"]
        predicted_mask = convert_reception(predicted_mask)
       
        ##mask = np.array(Image.open(mask))
        ##plt.imshow(mask)
        #st.image(Image.open(predicted_mask), width=None)
        
        st.header("Results")
        images = [Image.open(image), Image.open(mask), Image.open(predicted_mask)]
        captions = ["Input image", "Ground truth mask", "Predicted mask"]
        st.image(images, width=420, caption=captions)
    
if __name__=='__main__': 
    main()

