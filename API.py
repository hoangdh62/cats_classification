import flask
import cv2
import numpy as np
import ProcessImage as p
import tensorflow as tf
from flask import jsonify, request, make_response
from flask import send_file
from datetime import datetime
from keras.applications import ResNet50
from keras.layers.core import Flatten
from keras.layers import Input, Dense, Dropout
from keras.models import Model

def create_model(labels):
    baseModel = ResNet50(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(labels), activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    for layer in baseModel.layers:
        layer.trainable = False
    return model

app = flask.Flask(__name__)
app.config["DEBUG"] = True
# model = create_model(p.LABELS)
# model.load_weights('model/ep196-loss0.986-val_loss1.680.h5')
# graph = tf.get_default_graph()

@app.route('/Process', methods=['POST'])
def process():
    # data = request.get_json()
    # # print("JSON String " + str(data))
    # img = data['image']
    # print(img)
    if request.files.get("image"):
        now = datetime.now()
        time = now.strftime("%d%m%Y%H%M%S")
        img = request.files["image"]
        npimg = np.fromfile(img, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        graph.as_default()
        with graph.as_default():
            results = p.predict_image(model, img, p.LABELS)
        # cv2.imwrite("output/" + time + ".jpeg", img)
        # return send_file("result.txt")
        # response = make_response(results, 200)
        # response.mimetype = "text/plain"
        return results
@app.route('/home', methods=['POST'])
def home():
    if request.args.get("username") and request.args.get("password"):
        username = request.args.get("username")
        password = request.args.get("password")
        return
    else:
        return "None"

# app.run(host="192.168.1.219", port=8080,debug=False)
# app.run(host="192.168.1.14", port=5000,debug=False)
app.run()