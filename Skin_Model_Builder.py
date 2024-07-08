import tensorflow as tf
#from google.colab import drive
#drive.mount("./content/")
import cv2
import os
from sklearn.utils import resample
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    count = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        count += 1
        if img is not None:
            images.append([img,os.path.join(folder,filename)])
        if count > 10:
          break
    return images
def list_maker(folder):
  list1 = []
  for filename in os.listdir(folder):
    list1.append([filename[7:12],os.path.join(folder,filename)])
  return list1
#image_numbers_train = list_maker('./data/train')
#image_numbers_test = list_maker('./data/test')
#image_numbers[x][0] is the number of the image and image_numbers[x][1] is the filepath
import numpy as np
import pandas as pd

key = pd.read_csv(os.getcwd() + '/data/HAM10000_metadata.csv')
order = []
deletes = []
ids = []
count = 0
SIZE = 32

label=key['dx'].unique().tolist()  #Extract labels into a list

df_0 = key[key['dx'] == label[0]]
df_1 = key[key['dx'] == label[1]]
df_2 = key[key['dx'] == label[2]]
df_3 = key[key['dx'] == label[3]]
df_4 = key[key['dx'] == label[4]]
df_5 = key[key['dx'] == label[5]]
df_6 = key[key['dx'] == label[6]]

n_samples=500 
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

#Combined back to a single dataframe
key_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])

#Check the distribution. All classes should be balanced now.
image_paths = []
for i in key_balanced['image_id']:
    image_paths.append(os.path.join('./data/All Images/', i))
    #key_balanced['path'] = os.path.join('./data/All Images/', i)
key_balanced.insert(len(key_balanced.columns), "Paths", image_paths, True)
del image_paths

images = []
for i in key_balanced['Paths']:
    images.append(np.asarray(Image.open(i+'.jpg').resize((SIZE,SIZE))))
    #image_paths.append(os.path.join('./data/All Images/', i))
    #key_balanced['path'] = os.path.join('./data/All Images/', i)
key_balanced.insert(len(key_balanced.columns), "Images", images, True)
del images

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
labels = []
for i in key_balanced['dx']:
    labels.append(class_names.index(i))
    #image_paths.append(os.path.join('./data/All Images/', i))
    #key_balanced['path'] = os.path.join('./data/All Images/', i)
key_balanced.insert(len(key_balanced.columns), "Labels", labels, True)
del labels
#image_path = list_maker(r'C:\Users\Laith Qushair\.spyder-py3\Web_Page_Flask\data\All Images')

X = np.asarray(key_balanced['Images'].tolist())
X = X/255. # Normalize


Y=key_balanced['Labels']
#Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#Define the path and add as a new column
#key_balanced['path'] = key['image_id'].map(image_path)
#Use the path to read images.
#key_balanced['image'] = key_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE)))

#print(key[key.image_id=='ISIC_0033536'].dx)
#del count_test_dx_num
#import cv2
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
#train_datagen = ImageDataGenerator(rescale=1/255)

#img = mpimg.imread(image_numbers_test[0][1])
#print(type(img))
#print(img.shape)
#plt.imshow(img)
#divider = 3
#res = cv2.resize(img, dsize=(200, 150), interpolation=cv2.INTER_CUBIC)
#print(res.shape)
#plt.imshow(res)
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 450x600 with 3 bytes color
    # This is the first convolution
    #tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    # The second convolution
    #tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    #tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(32, activation='relu'),
    # Only 7 output neuron. For akiec, bcc, vasc, mel, df, nv, bkl
    tf.keras.layers.Dense(7, activation='softmax')
])
#from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#test = []
#start = 24306

batch_size = 16 
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])

'''
for i in range(len(image_numbers_test)):
  index = int(image_numbers_test[i][0])-start #IMPORTANT
  test.append([image_numbers_test[i][0],image_id_list[index][1]])
  if int(image_numbers_test[i][0]) != image_id_list[index][1]: #Ensures key has no gaps
    print("Wrong_Id")
print(test)
print(key.iloc[image_id_list[int(image_numbers_train[8][0]) - image_id_list[0][1]][0]])
'''
#Input row number from image_id_list to get df row
'''
img = img/255
start = image_id_list[0][1]
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #print(image_numbers_train[i][0])
    skin = mpimg.imread(image_numbers_train[i][1])
    skin = skin/255
    plt.imshow(skin, cmap=plt.cm.binary)
    plt.xlabel(key.iloc[image_id_list[int(image_numbers_train[i][0]) - start][0]].dx)
plt.show()
#reduce image size
#reduce data package
del key['dataset'],key['dx_type'],key['sex'],key['age']
#Reduce System RAM to run this line
train_images = []
train_labels = []
start = image_id_list[0][1]

dx_count = {'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0,
               'nv': 0, 'vasc': 0}

for i in range(len(image_numbers_train)):
  file_path = image_numbers_train[i][1]
  stripped_image_id = image_numbers_train[i][0]
  label = key.iloc[image_id_list[int(stripped_image_id) - start][0]].dx
  dx_count[label] += 1
  if dx_count[label] <= 600:
      skin = mpimg.imread(file_path)
      skin = cv2.resize(skin, dsize=(200, 150), interpolation=cv2.INTER_CUBIC)
      skin = skin/255
      train_images.append(skin)
      train_labels.append(label)
#Reduce System RAM to run this line
test_images = []
test_labels = []
start = image_id_list[0][1]
dx_count = {'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0,
               'nv': 0, 'vasc': 0}
del image_numbers_train

del order, deletes, i, skin, start, ids, image_id_list, image_numbers_test, img, key, list2, test
train_images = np.array(train_images)
train_labels = np.array(train_labels)
for i in range(len(train_labels)):
  train_labels[i] = class_names.index(train_labels[i])
train_labels = train_labels.astype(int)
model.fit(train_images, train_labels, epochs=10)

image_numbers_test = list_maker('./data/test')
key = pd.read_csv('./data/Skin_Data_Key.csv')
image_id_list = []
for i in range(len(key['image_id'])-1):
  image_id_list.append(int(key['image_id'][i][7:12]))
list2 = list(enumerate(image_id_list))
image_id_list = sorted(list2, key=lambda x: x[1])
start = image_id_list[0][1]
for i in range(len(image_numbers_test)):
  file_path = image_numbers_test[i][1]
  stripped_image_id = image_numbers_test[i][0]
  label = key.iloc[image_id_list[int(stripped_image_id) - start][0]].dx
  dx_count[label] += 1
  if dx_count[label] <= 300:
      skin = mpimg.imread(file_path)
      skin = cv2.resize(skin, dsize=(200, 150), interpolation=cv2.INTER_CUBIC)
      skin = skin/255
      test_images.append(skin)
      test_labels.append(label)

del train_images, train_labels
test_images = np.array(test_images)
test_labels = np.array(test_labels)
for i in range(len(test_labels)):
  test_labels[i] = class_names.index(test_labels[i])
test_labels = test_labels.astype(int)
score = model.evaluate(test_images, test_labels)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.summary())

#new_model = tf.keras.models.load_model('skin_model.keras')
#Loading Model
'''