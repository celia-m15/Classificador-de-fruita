# IMPORTS
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from old_school_methods import get_features
from upload_data_metrics import upload_data_aug, data_augmentation, upload_kaggle_data
from upload_data_metrics import give_accuracy, conf_matrix

rd = 42 # 0 , ...

# UPLOAD DATA
print('Uploading data...')
fruits = ['Apple', 'Banana', 'Lime', 'Orange', 'Pomegranate']
# fruits = ['Apple', 'Banana', 'Lime', 'Orange']
path = './Processed Images_Fruits/Good Quality_Fruits/'
df = upload_kaggle_data(path, fruits)

plt.imshow(df['image'][0])

# GET HOG FEATURES
print('Getting features...')
# df = get_features(df=df, features='hog_features')
df = get_features(df=df, features='color_histogram')
# df = get_features(df=df, features='lbp_features')

# TRAIN
print('Training...')
X = df.drop(['fruit'], axis=1)
y = df['fruit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rd)

# model = SVC(kernel = 'linear', gamma = 'scale', shrinking = False,)
model = RandomForestClassifier(max_depth=100, random_state=rd)
model.fit(X_train, y_train)

# PREDICT
print('Predict...')
y_pred = model.predict(X_test)

# ACCURACY
print('Accuracy...')
give_accuracy(y_test, y_pred, fruits)
    
# CONFUSION MATRIX
print('Confusion matrix...')
conf_matrix(y_test, y_pred, fruits)


print('PROVES AMB DADES PRÒPIES:')
# UPLOAD DATA
print('Uploading data...')
fruits = ['Apple', 'Banana', 'Orange', 'Lime']
path = './images_augm/'
df_a = upload_data_aug(path, fruits)
plt.imshow(df_a['image'][0])
df_a = data_augmentation(df_a)

# GET HOG FEATURES
print('Getting features...')
# df_a = get_features(df=df_a, features='hog_features')
df_a = get_features(df=df_a, features='color_histogram')
# df_a = get_features(df=df_a, features='lbp_features')

# TRAIN
print('Training...')
X_a = df_a.drop(['fruit'], axis=1)
y_a = df_a['fruit']
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.2, random_state=rd)
# model = SVC(kernel = 'linear', gamma = 'scale', shrinking = False,)
model = RandomForestClassifier(max_depth=100, random_state=rd)
model.fit(X_train_a, y_train_a)

# PREDICT
print('Predict...')
y_pred_a = model.predict(X_test_a)

# ACCURACY
print('Accuracy...')
give_accuracy(y_test_a, y_pred_a, fruits)
    
# CONFUSION MATRIX
print('Confusion matrix...')
conf_matrix(y_test_a, y_pred_a, fruits)



