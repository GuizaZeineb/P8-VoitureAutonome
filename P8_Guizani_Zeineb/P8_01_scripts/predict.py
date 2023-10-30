
import json
import pickle
import os



from numpy import argmax
import base64
from io import BytesIO


from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from azureml.core import Workspace, Run



from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

#SM_FRAMEWORK=tf.keras
#import segmentation_models as sm




import albumentations as A
#%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import segmentation_models as sm

import numpy as np

def init():
    """ La fonction init() va initialiser le service API au début, cette partie 
    ne sera pas exécutée à chaque fois il y a une nouvelle requête"""
    # Création de variables globales
    global model
    #global tokenizer

    # Indication du chemin du modèle
    model_path = os.getenv('AZUREML_MODEL_DIR')

    # Définition de l'architecture
    
    input_size=(128, 256, 3)
    BACKBONE = 'efficientnetb7' #BACKBONE = 'efficientnetb3' #BACKBONE = 'resnet34'
    # define network parameters
    n_classes = 8
    activation = 'softmax'
    #create model
    sm.set_framework('tf.keras')
    model = sm.Unet(BACKBONE,input_size, classes=n_classes, activation=activation)
    model.load_weights(model_path + '/outputs/unet.hdf5')


def pretraitement(original_j_img):
    #original_j_img = np.array(j_img_) # original_j_img = np.asarray(j_img_).astype(float32) #
    j_img = np.array(original_j_img) / 255
    resize = A.Resize(height=128, width=256, p=1)
    rcz = resize(image=j_img)
    msk = rcz['image']
    return msk



def cat2color(arr_to_convert):
    prediction_color = {0: (0,0,0), # void - background   "#1D507A"
                        1: (128,64,128), #flat - road  "#2F6999"
                        2:  (210, 190, 150), #(180,165,180), #construction  "#66A0D1"
                        3: (70,70,70), #object - poteau "#8FC0E9"
                        4: (152,251,152), #nature - vegetation "#4682B4"
                        5: (70,130,180), #sky  "#7f7f7f"
                        6: (255,0,0), #human  "#bcbd22"
                        7: (0,0,142) } #vehicle   "#9467bd"
    arr = np.zeros((*arr_to_convert.shape, 3))
    for k, v in prediction_color.items():
        arr[arr_to_convert == k] = v
    arr = arr.astype('uint8')
    return arr
    
def predict(msk):
    
    """Fonction qui va faire la prédiction des sentiments qui retourne le score puis
    suivant ce score elle va décoder le sentiment correspondant"""
       
    # Faire la prédiction
    val_preds = model.predict(np.expand_dims(msk, axis=0))  
    val_preds = np.squeeze(val_preds, axis=0)
    _mask = np.argmax(val_preds, axis=-1)
    predicted_mask = cv2.resize(_mask.astype(np.float32), (512, 256))
    # Vérification des couleurs
    pred = cat2color(predicted_mask)
    return pred


def convert_reception_flask(json_data):
    #Obtenir le json POSTÉ
    dict_data = json.loads(json_data) #Convertir json en dictionnaire

    img = dict_data["img"] #Sortez base64# str
    img = base64.b64decode(img) #Convertir les données d'image converties en base64 en données binaires d'origine# bytes
    img = BytesIO(img) # _io.Converti pour être géré par l'oreiller BytesIO
    return img

def convert_reception(img):
    #Obtenir l'image
    img = base64.b64decode(img) #Convertir les données d'image converties en base64 en données binaires d'origine# bytes
    img = BytesIO(img) # _io.Converti pour être géré par l'oreiller BytesIO
    return img

def convert_emission(img):
    #Convertir l'image d'oreiller en octets, puis en base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_byte = buffered.getvalue() # bytes
    img_base64 = base64.b64encode(img_byte) #octets encodés en base64 * pas str
    #C'est toujours des octets si json.Convertir en str en vidages(Parce que l'élément json ne prend pas en charge le type d'octets)
    img_str = img_base64.decode('utf-8') # str
    return img_str

    
def run(raw_data):
    """ Fonction appelée lors de l'appel à l'API à chaque nouvelle requête
    Dans cette fonction il y a récupération des données, application de la tockenisation,
    et le padding puis prédiction afin de décoder le sentiment"""
    try:
        
        #preds = []
        #data = json.loads(raw_data)["data"]
        
        
        #    Traitement pour récupérer l'image    #json_data = request.get_json() #Obtenir le json POSTÉ
        json_data = json.loads(raw_data)["img"]
        img = convert_reception(json_data)
        img = Image.open(img) 
        

        #________faire les prétraitements pour images        
        msk = pretraitement(img)
        pred = predict(msk)

   
        #img_str = convert_emission(pred)
        #msk = Image.fromarray(msk)
        pred = Image.fromarray((pred * 255).astype(np.uint8))

        img_str = convert_emission(pred)   #img_str = convert_emission(img)    

        
        
        #Renvoyer le résultat du traitement au client
        response  = {
            "text":"reponse- zeineb",
            "img":img_str #img_shape 
            }
        
        return json.dumps(response)#return json.dumps(preds) #return jsonify(response)



    except Exception as e:
        error = str(e)
        return error
