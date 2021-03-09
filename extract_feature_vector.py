from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from imutils import paths
from keras.models import load_model
from collections import deque
import numpy as np
import os
import cv2
from tqdm import tqdm
from scipy.spatial import distance

DATASET_PATH = "cat_data2/train"
LABELS = os.listdir(DATASET_PATH)

baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
# model.load_weights('model/ep101-60class.h5')

# initialize the video stream, pointer to output video file, and
# frame dimensions
features_vector1 = []
features_vector2 = []
filePaths = list(paths.list_images(DATASET_PATH))
class_test = ['Bombay','Havana']
for filePath in tqdm(filePaths):
	mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
	Q = deque(maxlen=128)
	true_label = filePath.split(os.path.sep)[-2]
	if true_label not in class_test:
		continue
	frame = cv2.imread(filePath)
	file_name = os.path.basename(filePath)
	# writer = None
	(W, H) = (None, None)

	print(filePath)
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	results = np.array(Q).mean(axis=0)
	if true_label == 'Bombay':
		features_vector1.append(results)
	elif true_label == 'Havana':
		features_vector2.append(results)

feature_vector1 = np.asarray([sum(x) for x in zip(*features_vector1)])/len(features_vector1)
feature_vector2 = np.asarray([sum(x) for x in zip(*features_vector1)])/len(features_vector2)
dst = distance.euclidean(feature_vector1, feature_vector2)
with open('report.txt', 'w') as f:
	f.write(str(feature_vector1)+'\n'+str(feature_vector2))
cv2.imshow(np.asarray(feature_vector1).reshape(64,32))
ggg=1

