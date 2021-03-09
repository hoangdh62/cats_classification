from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
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

DATASET_PATH = "cat_data/test"
LABELS = os.listdir(DATASET_PATH)

baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(LABELS), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

model.load_weights('model/ep196-loss0.986-val_loss1.680.h5')

# initialize the video stream, pointer to output video file, and
# frame dimensions
y_test = []
y_pred = []
filePaths = list(paths.list_images(DATASET_PATH))
for filePath in tqdm(filePaths):
	mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
	Q = deque(maxlen=128)
	frame = cv2.imread(filePath)
	file_name = os.path.basename(filePath)
	true_label = filePath.split(os.path.sep)[-2]
	y_test.append(true_label)
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

	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	results_sorted = sorted(range(len(results)), key=lambda j: results[j], reverse=True)[:5]
	# pred_label = LABELS[i]
	labels_sorted = [LABELS[results_sorted[0]], LABELS[results_sorted[1]], LABELS[results_sorted[2]]]
	if true_label in labels_sorted:
		pred_label = true_label
	else:
		pred_label = LABELS[i]
	# text = "{}".format(pred_label)
	# text = "["+LABELS[results_sorted[0]]+"-"+LABELS[results_sorted[1]]+"-"+LABELS[results_sorted[2]]+"]"
	# cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
	# 	1.25, (0, 255, 0), 5)

	y_pred.append(pred_label)

	# cv2.imwrite('cat_data/test_result_image/'+true_label+'_'+text+'_'+file_name, output)

array = confusion_matrix(y_test,y_pred)
print(sum(array[ii][ii] for ii in range(len(array)))/len(filePaths))

df_cm = pd.DataFrame(array, index=LABELS, columns=LABELS)
# plt.figure(figsize=(10,7))
df_norm_col = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]

sn.set(font_scale=0.4) # for label size
sn.heatmap(df_norm_col, annot=True, annot_kws={"size": 6}, cmap="Blues") # font size
plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
plt.savefig('cf_ep196-60class.png', pdi=300)
plt.show()