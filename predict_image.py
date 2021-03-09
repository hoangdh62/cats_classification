# USAGE
# python predict_video.py --model model/activity.model --label-bin model/lb.pickle --input example_clips/lifting.mp4 --output output/lifting_128avg.avi --size 128
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
# import the necessary packages
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
	help="path to trained serialized model", default='model/ep101-60class.h5')
ap.add_argument("-l", "--label-bin", required=False,
	help="path to  label binarizer", default='model/lb.pickle')
ap.add_argument("-i", "--input", required=False,
	help="path to our input video", default='image_test/panther.jpg')
ap.add_argument("-o", "--output", required=False,
	help="path to our output video", default='output/walk_196.avi')
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# lb = pickle.loads(open(args["label_bin"], "rb").read())
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

model.load_weights('model/ep101-60class.h5')

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
frame = cv2.imread(args["input"])
# writer = None
(W, H) = (None, None)

# if the frame dimensions are empty, grab them
if W is None or H is None:
	(H, W) = frame.shape[:2]

output = frame.copy()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (224, 224)).astype("float32")
frame -= mean

# make predictions on the frame and then update the predictions
# queue
preds = model.predict(np.expand_dims(frame, axis=0))[0]
Q.append(preds)

# perform prediction averaging over the current history of
# previous predictions
results = np.array(Q).mean(axis=0)
i = np.argmax(results)
# label = lb.classes_[i]
label = LABELS[i]
# label = "Stand"
print(label, max(results), results)

# draw the activity on the output frame
text = "{}".format(label+str(max(results)))
cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
	1.25, (0, 255, 0), 5)

# check if the video writer is None
# if writer is None:
# 	# initialize our video writer
# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# 	writer = cv2.VideoWriter(args["output"], fourcc, 30,
# 		(W, H), True)
cv2.imshow('image',output)
cv2.waitKey(0)
cv2.destroyAllWindows()
# write the output frame to disk
cv2.imwrite('image_test/output/test6_output.jpg', output)

# release the file pointers
print("[INFO] cleaning up...")
# writer.release()
# vs.release()