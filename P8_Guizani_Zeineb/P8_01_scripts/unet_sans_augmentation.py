
import warnings
import cv2
import matplotlib.pyplot as plt
from azureml.core import Run
from tensorflow import keras
import os
import numpy as np
from random import shuffle
import PIL
from PIL import Image
import albumentations as A
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend  # as keras
import segmentation_models as sm
from segmentation_models.metrics import iou_score
from tensorflow.keras.losses import CategoricalCrossentropy
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,  RandomGamma, Blur, ElasticTransform, Emboss
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, Callback


# Importation des librairies
# Disable warnings in Anaconda
warnings.filterwarnings('ignore')


# ---->  Génère erreur dans le script non connu par azure %matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


# ---->  Génère erreur dans le script non connu par azure %env SM_FRAMEWORK=tf.keras
#import tensorflow as tf
# sm.set_framework('tf.keras')


#from keras.losses import CategoricalCrossentropy


parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='')

# parser.add_argument("--val_folder", type=str, dest="val_folder",
#                   help="folder where validation data are stored")
# parser.add_argument("--train_folder", type=str, dest="train_folder",
#                   help="folder where training data are stored")


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


# def Unet(pretrained_weights = None,input_size = (256,256,1)):
def Unet(pretrained_weights=None, input_size=(128, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
#    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
# ____________>>> ______Passer de 64 à 8 directement la sortie c'est 8 pixels suivant les classes_______
#    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = Conv2D(8, 1, activation='softmax')(conv9)


#    model = tf.keras.Model(input = inputs, output = conv10)
    model = Model(inputs, conv10)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
#    return conv10


#from matplotlib import pyplot as plt

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


def main():

    args = parser.parse_args()
    data_folder = args.data_folder
    print('Data folder:', data_folder)

    # Création du répartoire outputs por stocker les modèles finaux
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    run = Run.get_context()

    resize = A.Resize(height=128, width=256, interpolation=3, p=1)

    AUGMENTATIONS_TRAIN = Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        Emboss(p=0.25),
        Blur(p=0.25, blur_limit=7),
        ElasticTransform(p=0.25, alpha=120, sigma=120 *
                         0.05, alpha_affine=120 * 0.03)

    ])

    BATCH_SIZE = 4
    train_data_dir = os.path.join(data_folder, 'train')
    train_generator = Dataloder(train_data_dir, resize, BATCH_SIZE, False)
    #train_generator =  Dataloder(train_data_dir, resize, BATCH_SIZE, augmentation=AUGMENTATIONS_TRAIN)

    val_data_dir = os.path.join(data_folder, 'val')
    val_generator = Dataloder(val_data_dir, resize, BATCH_SIZE, False)

    model = Unet(input_size=(128, 256, 3))

    # We don't use segmentation models for Unet
    #model = Unet(backbone_name = 'efficientnetb0', encoder_weights='imagenet', encoder_freeze = False)

    loss = CategoricalCrossentropy()

    model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=[iou_score])

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

    # mode_autosave = ModelCheckpoint("./outputs/checkpoint",
    # monitor='val_iou_score',
    # verbose=1, save_best_only=True,
    # save_weights_only=True, mode='max'
    # )

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
