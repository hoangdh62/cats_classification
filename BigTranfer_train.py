#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print (os.path.abspath(os.curdir))
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import image

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from PIL import Image
import requests
from io import BytesIO


################################################################################################################################
#                                                CREATING DATA

base_path = "E:/AI Intern/Flower_Classification Res/Flickr"
# categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
categories = ['daisy', 'dandelion', 'dandelion', 'foxglove', 'lilac', 'lilies', 'lupin', 'myosotis', 'orchid', 'pansy', 'rose', 'sunflower', 'tulip']
# 13 items


# In[2]:


#@title Plotting helper functions (hidden)
#@markdown Credits to Xiaohua Zhai, Lucas Beyer and Alex Kolesnikov from Brain Zurich, Google Research

# Show the MAX_PREDS highest scoring labels:
MAX_PREDS = 13
# Do not show labels with lower score than this:
MIN_SCORE = 0.5 

def show_preds(logits, image, correct_flowers_label=None, tf_flowers_logits=False):
    if len(logits.shape) > 1:
        logits = tf.reshape(logits, [-1])
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), squeeze=False)
    ax1, ax2 = axes[0]
    ax1.axis('off')
    ax1.imshow(image)
    if correct_flowers_label is not None:
        ax1.set_title(tf_flowers_labels[correct_flowers_label])
    classes = []
    scores = []
    logits_max = np.max(logits)
    softmax_denominator = np.sum(np.exp(logits - logits_max))
    for index, j in enumerate(np.argsort(logits)[-MAX_PREDS::][::-1]):
        score = 1.0/(1.0 + np.exp(-logits[j]))
        if score < MIN_SCORE: break
        if not tf_flowers_logits:
      # predicting in imagenet label space
            classes.append(imagenet_int_to_str[j])
    else:
      # predicting in tf_flowers label space
        classes.append(tf_flowers_labels[j])
        scores.append(np.exp(logits[j] - logits_max)/softmax_denominator*100)

    ax2.barh(np.arange(len(scores)) + 0.1, scores)
    ax2.set_xlim(0, 100)
    ax2.set_yticks(np.arange(len(scores)))
    ax2.yaxis.set_ticks_position('right')
    ax2.set_yticklabels(classes, rotation=0, fontsize=14)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.set_xlabel('Prediction probabilities', fontsize=11)


# In[3]:


#@title Helper functions for loading image (hidden)

def preprocess_image(image):
    image = np.array(image)
    # reshape into shape [batch_size, height, width, num_channels]
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)  
    return image

def load_image_from_url(url):
    """Returns an image with shape [1, height, width, num_channels]."""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = preprocess_image(image)
    return image


# In[4]:


# # Load image (image provided is CC0 licensed)
# img_url = "https://i.pinimg.com/originals/17/a5/d1/17a5d17b2f30c1ad24b6fd182dc89c59.jpg"
# image = load_image_from_url(img_url)

# # Run model on image
# logits = module(image)

# # Show image and predictions
# show_preds(logits, image[0])


# In[ ]:





# In[5]:



#Load file names
fnames = []
for category in categories:
    flower_folder = os.path.join(base_path, category)
    file_names = os.listdir(flower_folder)
    full_path = [os.path.join(flower_folder, file_name) for file_name in file_names]
    fnames.append(full_path)

#Load images
images = []
for names in fnames:
    one_category_images = [cv2.imread(name)  for name in names if (cv2.imread(name)) is not None]
    images.append(one_category_images)

print('\nNumber of images for each category:', [len(f) for f in images])

#Calculate the minimal shape for all images
for i,imgs in enumerate(images):
    shapes = [img.shape for img in imgs]
    widths = [shape[0] for shape in shapes]
    heights = [shape[1] for shape in shapes]
    #print('%dx%d is the  shape for %s' % (np.min(widths), np.min(heights), categories[i]))
    
def cvtRGB(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)


# In[6]:


img_width, img_height = 224, 224

img = images[1][1]
print(img.shape)
resized_img = resize(img, (img_width, img_height, 3))
resized_img2 = cv2.resize(img,(img_width, img_height), interpolation = cv2.INTER_CUBIC)
print(resized_img.shape)

# Apply resize to all images
resized_images = []
for i,imgs in enumerate(images):
    resized_images.append([cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC) for img in imgs])

train_images = []
test_images = []
for imgs in resized_images:
    train, test = train_test_split(imgs, train_size=0.9, test_size=0.1)
    train_images.append(train)
    test_images.append(test)

len_train_images = [len(imgs) for imgs in train_images]
print(len_train_images)
print('sum of train images:', np.sum(len_train_images))
train_categories = np.zeros((np.sum(len_train_images)), dtype='uint8')
for i in range(5):
    if i is 0:
        train_categories[:len_train_images[i]] = i
    else:
        train_categories[np.sum(len_train_images[:i]):np.sum(len_train_images[:i+1])] = i
        
len_test_images = [len(imgs) for imgs in test_images]
print(len_test_images)
print('sum of test_images:', np.sum(len_test_images))
test_categories = np.zeros((np.sum(len_test_images)), dtype='uint8')
for i in range(5):
    if i is 0:
        test_categories[:len_test_images[i]] = i
    else:
        test_categories[np.sum(len_test_images[:i]):np.sum(len_test_images[:i+1])] = i


# In[7]:


#Convert image data to numpy array
#Convert class labels to binary class labels
print("\nConvert class labels to binary class labels")

tmp_train_imgs = []
tmp_test_imgs = []
for imgs in train_images:
    tmp_train_imgs += imgs
for imgs in test_images:
    tmp_test_imgs += imgs
train_images = np.array(tmp_train_imgs)
test_images = np.array(tmp_test_imgs)

print('Before converting')
print('train data:', train_images.shape)
print('train labels:', train_categories.shape)

train_data = train_images.astype('float16')
test_data = test_images.astype('float16')
train_labels = np_utils.to_categorical(train_categories, len(categories))
test_labels = np_utils.to_categorical(test_categories, len(categories))
print()
print('After converting')
print('train data:', train_data.shape)
print('train labels:', train_labels.shape)


# In[8]:


# seed = 1000
# np.random.seed(seed)
# np.random.shuffle(train_data)
# np.random.seed(seed)
# np.random.shuffle(train_labels)
# np.random.seed(seed)
# np.random.shuffle(test_data)
# np.random.seed(seed)
# np.random.shuffle(test_labels)

# # train_data = train_data[:3400]
# # train_labels = train_labels[:3400]
# # test_data = test_data[:860]
# # test_labels = test_labels[:860]
# print('shape of train data:', train_data.shape)
# print('shape of train labels:', train_labels.shape)
# print('shape of val data:', test_data.shape)
# print('shape of val labels:', test_labels.shape)


# In[15]:


# Load model into KerasLayer
# model_url = 'https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1'
model_url = "https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1"
module = hub.KerasLayer(model_url)

model = Sequential([
    module,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(categories), activation='softmax')])

# model.trainable = True


# In[10]:


# batch_size = 32
# SCHEDULE_LENGTH = SCHEDULE_LENGTH * 32 / batch_size
# epochs = 30

# def cast_to_tuple(features):
#     return (features['image'], features['label'])
  
# def preprocess_train(features):
#   # Apply random crops and horizontal flips for all tasks 
#   # except those for which cropping or flipping destroys the label semantics
#   # (e.g. predict orientation of an object)
#     features['image'] = tf.image.random_flip_left_right(features['image'])
#     features['image'] = tf.image.resize(features['image'], [RESIZE_TO, RESIZE_TO])
#     features['image'] = tf.image.random_crop(features['image'], [CROP_TO, CROP_TO, 3])
#     features['image'] = tf.cast(features['image'], tf.float32) / 255.0
#     return features

# def preprocess_test(features):
#     features['image'] = tf.image.resize(features['image'], [RESIZE_TO, RESIZE_TO])
#     features['image'] = tf.cast(features['image'], tf.float32) / 255.0
#     return features


# pipeline_train = (train_data
#                   .shuffle(10000)
#                   .repeat(int(SCHEDULE_LENGTH * batch_size / len(categories) * epochs) + 1 + 50)  # repeat dataset_size / num_steps
#                   .map(preprocess_train, num_parallel_calls=8)
#                   .batch(batch_size).map(cast_to_tuple)  # for keras model.fit
#                   .prefetch(2))

# pipeline_test = (test_data.map(preprocess_test, num_parallel_calls=1)
#                   .map(cast_to_tuple)  # for keras model.fit
#                   .batch(BATCH_SIZE).prefetch(2))


# In[16]:


batch_size = 32
####################################################### Create generator ImageDataGenerator Augmention
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.4,height_shift_range=0.4,shear_range=0.2,zoom_range=0.3,horizontal_flip=True)
# Note that the test data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255,)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow(train_data,train_labels,batch_size=batch_size)
test_generator = test_datagen.flow(test_data,test_labels,batch_size=batch_size)


# In[ ]:


# Define optimiser and loss
# RESIZE_TO = 512
# RESIZE_TO = 480

# SCHEDULE_LENGTH = 500
SCHEDULE_BOUNDARIES = [200, 300, 400]
lr = 0.003 
# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.          
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES,values=[lr, lr*0.1, lr*0.001, lr*0.0001])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
####Train the model , batch_size=batch_size   SparseCategoricalCrossentropy , steps_per_epoch=10
start = time.time()

model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])
history = model.fit(train_generator, 
                    epochs= 50, 
                    validation_data=test_generator)

end = time.time()
duration = end - start
print ('\n model_ResNet50 took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs) )


# In[ ]:


# Plots for training and testing process: loss and accuracy

def plot_model_history(model_name, history, epochs):

    print(model_name)
    plt.figure(figsize=(15, 5))

    # summarize history for accuracy
    plt.subplot(1, 2 ,1)
    plt.plot(np.arange(0, len(history['acc'])), history['acc'], 'r')
    plt.plot(np.arange(1, len(history['val_acc'])+1), history['val_acc'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Accuracy vs. Test Accuracy')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'test'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['loss'])+1), history['loss'], 'r')
    plt.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Loss vs. Test Loss')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

plot_model_history('Biểu đồ gì gì đó', history.history, epochs)


# In[ ]:


####################################################### PREDICT
for i in range(0, 37):
    img_path = 'E:/AI Intern/Flower_Classification Res/Unlabel/%s.jpg' % (i)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    pred = model.predict(x)
    percent = np.max(pred)
    label = int(np.argmax(pred)) 

    print(pred)
    print("%s(%s): %.2f" % (categories[label], label, percent*100.))

    plt.imshow(image.load_img(img_path))
    plt.show()

import gc
gc.collect()

