

import requests
import json
from azureml.core import Workspace  , Dataset
import numpy as np
import os
from io import BytesIO
import base64
from PIL import Image
#from werkzeug.utils import secure_filename
from flask import Flask, request,  jsonify



#app = Flask(__name__)
app = Flask(__name__, static_folder='/static')
path_to_static = "static/images/loaded"




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


def convert_emission(img):
    #img = Image.open(img)
    # Convertir l'image d'oreiller en octets, puis en base64

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_byte = buffered.getvalue()  # bytes
    # octets encodés en base64 * pas str
    img_base64 = base64.b64encode(img_byte)
    # C'est toujours des octets si json.Convertir en str en vidages(Parce que l'élément json ne prend pas en charge le type d'octets)
    img_str = img_base64.decode('utf-8')  # str
    return img_str

def get_dstore():
        ws = Workspace.get(name='MLprojet8', 
                           subscription_id='c3b35390-a141-477b-b971-cd4e8b57d43a', 
                           resource_group='OC_p8')

        datastore = ws.get_default_datastore()
        #ds_paths = [(datastore, 'preprocessed/val/')]
        return datastore


@app.route("/")
#@app.route('/', methods=['GET'])
def hello():
    return "Welcome to machine learning model APIs!"

@app.route("/array")
def fect_array():
    a = numpy.array([1, 2, 3.5])
    return  str(a )


@app.route('/api/get_img_list/')
def get_img_from_numpy():
    """Récupération de la liste des images disponibles depuis un fichier numpy

    Returns:
        [type]: [description]
    """
    val_files = np.load('static/data/val_files.npy')
    #val_files = np.load(os.path.join(basedir, 'val_files.npy'))
    print("val_files", val_files)
    print('-- réponse envoyée au client')
    response = {'status': 'ok', 'data': val_files.tolist()}
    return jsonify(response)
    # return json.dumps(response)
    
    
@app.route("/api/get_images", methods=["Get", "Post"])
def get_image_mask():
    if request.method == "POST":
        data = request.data
        id_image = json.loads(data)["data"]


        datastore = get_dstore()
        img_name =  id_image + "_leftImg8bit.png"
        mask_name = id_image + "_gtFine_labelIds.png"

        img_azure_path  =  'preprocessed/val/' + img_name
        mask_azure_path =  'preprocessed/val/' + mask_name


        #___Download image__
        img_path =  [(datastore, img_azure_path)]
        raw_img = Dataset.File.from_files(path=img_path)
        path_to_static = "static//images//loaded" #os.path.join (project_folder,"static//images//loaded")
        raw_img.download(path_to_static, overwrite=True)

        #___Download mask__
        mask_path =  [(datastore, mask_azure_path)]
        raw_mask = Dataset.File.from_files(path=mask_path)
        raw_mask.download(path_to_static, overwrite=True)

        img_path_local = os.path.join(path_to_static,img_name)
        mask_path_local = os.path.join(path_to_static,mask_name)

        img = Image.open(img_path_local)        

        
        
        #img = Image.open(j_img_)
        img_str = convert_emission(img)

        mask = Image.open(mask_path_local)
        
        mask = cat2color(np.array(mask))
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        # L inintiale la bonne
        mask_str = convert_emission(mask)

        #img_str = convert_emission(j_img_)
        #mask_str = convert_emission(j_mask_)
        files = {
            "img": img_str,
            "mask": mask_str
        }
    return json.dumps(files)


@app.route('/api/prediction', methods=["Get", "Post"])
def mask_prediction():
    if request.method == "POST":
        data = request.data
        id_image = json.loads(data)["data"]

        img_name = id_image + "_leftImg8bit.png"
        #______________> Path Flask Local
        #img_path_local = os.path.join(path_to_static,img_name)
        j_img_ = os.path.join(path_to_static,img_name)

        img = Image.open(j_img_)
        img_str = convert_emission(img)

        files = {
            "text": "envoi",
            "img": img_str
        }

        input_data = json.dumps(files)

        # Set the content type
        headers = {'Content-Type': 'application/json',
                   'Cache-Control': 'no-cache'}
        service = "http://4c568a7f-267d-4cac-b079-94dbdb4f76a3.westeurope.azurecontainer.io/score"
        # Make the request to azure machine learning model and display the response
        # Make the call using post
        # POST sur le serveur en tant que jso
        resp = requests.post(service, input_data, headers=headers)
        mask = json.loads(resp.json())["img"]
    return json.dumps({"status": "ok", "img": mask})

if __name__ == '__main__':
    app.run(debug=True)


# kill -9 `lsof -i:5000 -t` to kill the process using the port
