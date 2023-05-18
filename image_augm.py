import pandas as pd
import numpy as np
import cv2
import os
import random
from PIL import Image
from deep_learning_methods import ImageDataset
from torch.utils.data import DataLoader

def colorjitter(img, cj_type="b"):
    '''
    ### Different Color Jitter , es pot canviar saturacio, llum o contrast###
    img: image
    cj_type: {b: brightness, c: contrast}
    '''
    if cj_type == "b":
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
        
    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img
 
def noisy(img, noise_type="sp"):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    if noise_type == "gauss":
        image=img.copy() 
        mean=0
        st=0.7
        gauss = np.random.normal(mean,st,image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image,gauss)
        return image
    
    elif noise_type == "sp":
        image=img.copy() 
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image
    
def rotate_and_scale_image(image, angle, scale):
    height, width = image.shape[:2]

    # Calcular el centre de la imatge
    center = (width // 2, height // 2)

    # Definir la matriu de transformació
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Aplicar la transformació a la imatge
    rotated_image = cv2.warpAffine(image, M, (width, height))

    return rotated_image


def upload_data_aug(path: str, fruits: list):
    cols = ['fruit', 'image']
    df = pd.DataFrame(columns=cols)
    images = []
    labels = []
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(image, dsize=(256, 256)))
        mess = filename.split("_")
        labels.append(mess[0])
    df['image'] = images
    df['fruit'] = labels
    return df.reset_index()

def from_pd_to_dataloader(df):
    images = []
    for i, row in df.iterrows():
        # load the image from the file path in the 'image' column
        image = row['image'].copy()
        # convert the image to a numpy array
        image = np.array(image)
        # normalize the pixel values to be between 0 and 1
        image = image / 255.0
        # add the image to the list of preprocessed images
        images.append(image)

    # convert the list of images to a numpy array
    images = np.array(images)

    # get the labels as a numpy array
    labels = np.array(df['fruit'])
    label_to_idx = {label: idx for idx, label in enumerate(set(labels))}
    train_labels = np.array([label_to_idx[label] for label in labels])
 
    test_d = ImageDataset(images, train_labels)
    test_dl = DataLoader(test_d, batch_size=32, shuffle=False)
    return test_dl

def data_augmentation(df):
    cols = ['fruit', 'image']
    #data augm
    images = []
    labels = []
    df_fruit = pd.DataFrame(columns = cols)
    for img, fruit in zip(df['image'], df['fruit']):
        #colorjitter
        list_type = ['c']*3
        list_type.extend(['b']*3)
        for type_n in list_type:
            new_img = colorjitter(img, type_n)
            images.append(new_img)
            labels.append(fruit)
        
        #noisy
        list_type = ['gauss']*3
        list_type.extend(['sp']*3)
        for type_n in list_type:
            new_img = noisy(img, noise_type=type_n)            
            images.append(new_img)
            labels.append(fruit)
  
        #rotate and scale
        for i in range(5):
            angle = random.randint(20, 180)
            new_img = rotate_and_scale_image(img, angle, 1.5)
            images.append(new_img)
            labels.append(fruit)

    df_fruit['image'] = images
    df_fruit['fruit'] = labels
    df_fruit.reset_index()
    df = pd.concat([df, df_fruit])
    df_new = df.reset_index()
    
    test_dl = from_pd_to_dataloader(df_new)
    
    return test_dl
