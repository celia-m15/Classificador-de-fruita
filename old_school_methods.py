# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:48:39 2023

@author: alexc
"""

# IMPORTS
import pandas as pd
import cv2
import skimage
from skimage.feature import hog


# HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
def extract_hog_features(image):
    return hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, channel_axis=-1)


def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_lbp_features(image):
    gray = skimage.color.rgb2gray(image)
    lbp = skimage.feature.local_binary_pattern(gray, P=8, R=1)
    return lbp.flatten()


def get_features(df: pd.DataFrame, features: str='hog_features'):
    # GET HOG FEATURES
    if features == 'hog_features':
        df[features] = df['image'].apply(lambda x: extract_hog_features(x))
    elif features == 'color_histogram':
        df[features] = df['image'].apply(lambda x: extract_color_histogram(x))
    elif features == 'lbp_features':
        df[features] = df['image'].apply(lambda x: extract_lbp_features(x))
    
    # make each position of the array a column of the df
    hog_df = pd.DataFrame(df[features].values.tolist())
    hog_df.columns = ['feature{}'.format(i) for i in range(hog_df.shape[1])]
    df = pd.concat([df, hog_df], axis=1)
    df = df.drop([features, 'image', 'index'], axis=1)
    df = df.dropna(axis=1)
    return df
