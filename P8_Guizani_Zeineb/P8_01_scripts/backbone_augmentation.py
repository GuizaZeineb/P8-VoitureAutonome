


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback
from albumentations import (
    Compose, HorizontalFlip,RandomBrightnessContrast, RandomGamma,
        Emboss, Blur, ElasticTransform
)
from tensorflow.keras.losses import CategoricalCrossentropy

import segmentation_models as sm
#__________________
from tensorflow.keras import backend  # as keras
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
#____________


import albumentations as A
from PIL import Image
import PIL
from random import shuffle
import numpy as np
import os
from tensorflow import keras
from azureml.core import Run
import matplotlib.pyplot as plt
import cv2
import warnings

from segmentation_models.metrics import iou_score
#from segmentation_models import Unet
# Importation des librairies
# Disable warnings in Anaconda
warnings.filterwarnings('ignore')




parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='')


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and build batches

    Args:
        data_folder: str path to the data.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, data_folder, resize, batch_size=1, shuffle=False, augmentation=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = list(set(["_".join(path.split('_')[:3])
                                 for path in os.listdir(data_folder)]))
        self.indexes = [os.path.join(data_folder, path)
                        for path in self.indexes]
        self.resize = resize
        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size

        images = []
        masks = []
        for j in range(start, stop):
            root_file = self.indexes[j]
            image_file = root_file + "_leftImg8bit.png"
            mask_file = root_file + "_gtFine_labelIds.png"
            image = np.array(Image.open(image_file))
            mask = np.array(Image.open(mask_file))
            sample = self.resize(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            # apply augmentations
            if self.augmentation:
                # . ----L'augmentation initiale
                #sample = self.augmentation(image=image, mask=mask)
                #sample = aug_with_crop(image_size = 1024)(image=image, mask=mask)
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # divide by 255 to normalize images from 0 to 1
            image = image/255.
            images.append(image)
            masks.append(keras.utils.to_categorical(mask, num_classes=8))
            # masks.append(mask)

        # transpose list of lists
        image_batch = np.stack(images, axis=0)
        mask_batch = np.stack(masks, axis=0)

        return image_batch, mask_batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)



def plot_history(history):
    iou = history.history['iou_score']
    val_iou = history.history['val_iou_score']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    plt.savefig("./curves.png")

    max_value = max(val_iou)  # Return the max value of the list.
    max_index = val_iou.index(max_value)  # Find the index of the max value.
    print("max_index: ", max_index)
    dict_metrics = {
        'iou': max_value,
        'val_iou': val_iou[max_index],
        'loss': loss[max_index],
        'val_loss':  val_loss[max_index]
    }
    # Print Best metrics
    for item in dict_metrics.items():
        print(item)
    print ('these are backbone_diceloss_augmentation results')


def main():

    args = parser.parse_args()
    data_folder = args.data_folder
    print('Data folder:', data_folder)

    # Création du répartoire outputs por stocker les modèles finaux
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    run = Run.get_context()

    resize = A.Resize(height=128, width=256, p=1)

    AUGMENTATIONS_TRAIN = Compose([
        #HorizontalFlip(p=0.5),
        #RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        #Emboss(p=0.25),
        Blur(p=0.25, blur_limit=7) #,
        #ElasticTransform(p=0.25, alpha=120, sigma=120 *
        #                 0.05, alpha_affine=120 * 0.03)

    ])

    BATCH_SIZE = 4
    train_data_dir = os.path.join(data_folder, 'train')
    #train_generator = Dataloder(train_data_dir, resize, BATCH_SIZE, False)
    train_generator =  Dataloder(train_data_dir, resize, BATCH_SIZE, augmentation=AUGMENTATIONS_TRAIN)


    val_data_dir = os.path.join(data_folder, 'val')
    val_generator = Dataloder(val_data_dir, resize, BATCH_SIZE, False)


    # define network parameters
    BACKBONE = 'resnet34' #BACKBONE = 'efficientnetb3'
    n_classes = 8
    activation = 'softmax'
    #input_size=(128, 256, 3)

    #create model  
    sm.set_framework('tf.keras')
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    #model = Unet(BACKBONE, classes=n_classes, activation=activation) #, encoder_weights='imagenet'


    loss = sm.losses.dice_loss #loss = CategoricalCrossentropy()
    #metrics=[sm.metrics.iou_score]
    # compile keras model with defined optimozer, loss and metrics

    #model.compile('Adam', loss, metrics)
    model.compile(optimizer = Adam(lr = 1e-3), loss=loss, metrics = [iou_score])
    


    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log('loss', log['loss'])
            run.log('val_loss', log['val_loss'])
            run.log('iou_score', log['iou_score'])
            run.log('val_iou_score', log['val_iou_score'])

    # reduces learning rate on plateau
    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                   cooldown=10,
                                   patience=10, verbose=1,
                                   min_lr=0.1e-5)
    # model autosave callbacks

    #mode_autosave = ModelCheckpoint("./outputs/checkpoint",
                                    #monitor='val_iou_score',
                                    #verbose=1, save_best_only=True,
                                    #save_weights_only=True, mode='max'
                                    #)

    model_checkpoint = ModelCheckpoint('./outputs/unet.hdf5', 
                                       monitor='val_iou_score', 
                                       verbose=1, save_best_only=True, 
                                       save_weights_only=True, mode='max')

    # stop learining as metric on validatopn stop increasing
    early_stopping = EarlyStopping(patience=10, verbose=1, mode='auto')

    # tensorboard for monitoring logs
    tensorboard = TensorBoard(log_dir='./logs/tenboard', histogram_freq=0,
                              write_graph=True, write_images=False)

    # Enregistrement de tous les détails nécessaires dans le callback
    callbacks = [model_checkpoint, lr_reducer,
                 tensorboard, early_stopping, LogRunMetrics()]
    #callbacks = [mode_autosave, lr_reducer, tensorboard, early_stopping, LogRunMetrics()]

    history = model.fit(train_generator, shuffle=True,
                        epochs=30, use_multiprocessing=False,  # epochs = 50, workers=4
                        validation_data=val_generator,
                        verbose=1, callbacks=callbacks)

    # Pour afficher l'évolution des performances en fonction de l'epoc
    plot_history(history)

    run.log_image(name='loss/iou curves', path='./curves.png')
    run.complete()


if __name__ == "__main__":
    main()
