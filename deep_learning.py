# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:12:37 2023

@author: alexc
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

from deep_learning_methods import ImageDataset, CNNModel
from upload_data_metrics import upload_data_aug, data_augmentation, upload_kaggle_data
from upload_data_metrics import conf_matrix, give_accuracy

"""
assert torch.cuda.is_available(), "GPU is not enabled"
# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""


# UPLOAD DATA
print('Uploading data...')

fruits = ['Apple', 'Banana', 'Lime', 'Orange', 'Pomegranate']
path = './Processed Images_Fruits/Good Quality_Fruits/'
data = upload_kaggle_data(path, fruits)
"""
fruits = ['Apple', 'Banana', 'Orange', 'Lime']
path = './images_augm/'
df = upload_data_aug(path, fruits)
data = data_augmentation(df)
"""

im_size = 256

images = []
for i, row in data.iterrows():
    # load the image from the file path in the 'image' column
    image = row['image'].copy()
    # resize the image to 256x256 pixels
    image = np.array(Image.fromarray(image).resize((im_size, im_size)))
    # convert the image to a numpy array
    image = np.array(image)
    # normalize the pixel values to be between 0 and 1
    image = image / 255.0
    # add the image to the list of preprocessed images
    images.append(image)

# convert the list of images to a numpy array
images = np.array(images)

# get the labels as a numpy array
labels = np.array(data['fruit'])

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# convert the labels to integers
label_to_idx = {label: idx for idx, label in enumerate(set(labels))}
train_labels = np.array([label_to_idx[label] for label in train_labels])
test_labels = np.array([label_to_idx[label] for label in test_labels])

# create custom datasets and data loaders for the train and test sets
train_dataset = ImageDataset(train_images, train_labels)
test_dataset = ImageDataset(test_images, test_labels)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 32
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 32

model = CNNModel(num_labels=len(fruits), image_size=im_size)

# set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
print('Loading the model...')
model.load_state_dict(torch.load('cnn_model.pth'))
# train the CNN model
"""
print('Training the model...')
num_epochs = 7
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs, im_size)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {running_loss/len(train_dataloader):.5f}")

torch.save(model.state_dict(), 'cnn_model.pth')


# predict on the test set
print('Predicting...')
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for data in test_dataloader:
        images, labels = data
        outputs = model(images, im_size)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

print('Accuracy on test set: %d %%' % (100 * correct / total))
inv_map = {v: k for k, v in label_to_idx.items()}
y_true_str = [inv_map[y] for y in y_true]
y_pred_str = [inv_map[y] for y in y_pred]

# ACCURACY
give_accuracy(np.array(y_true_str), np.array(y_pred_str), fruits)

# CONFUSION MATRIX
print('Confusion matrix...')
conf_matrix(y_true_str, y_pred_str, fruits)
