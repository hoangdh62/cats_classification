import wikipediaapi
from collections import deque
import numpy as np
import cv2
import os
file = open('labels.txt','r')
LABELS = file.read().split('\n')
wiki = wikipediaapi.Wikipedia('en')

def wiki_search(labels_sorted):
    dict = {'top1':{
                    'image_url':'',
                    'content':''},
            'top2': {
                'image_url': '',
                'content': ''},
            'top3':{
                    'image_url':'',
                    'content':'' }
            }
    for i in range(len(labels_sorted)):
        try:
            inf = wiki.page(labels_sorted[i]+' cat')
            dict[f'top{i+1}'].update({'image_url':inf.images[0],'content':inf.content})
        except:
            try:
                inf = wiki.page(labels_sorted[i])
                dict[f'top{i+1}'].update({'image_url':inf.images[0],'content':inf.content})
            except:
                dict[f'top{i+1}'].update({'image_url':'Not found URL','content':'Not found info'})
    return dict

def predict_image(model, img, LABELS):

    # initialize the image mean for mean subtraction along with the
    # predictions queue
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen=128)

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    frame = img
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
    results_sorted = sorted(range(len(results)), key=lambda j: results[j], reverse=True)[:3]
    # labels_sorted = [LABELS[results_sorted[0]], LABELS[results_sorted[1]], LABELS[results_sorted[2]]]
    labels_sorted = LABELS[results_sorted[0]]+': '+str(results[results_sorted[0]])+'\n'+LABELS[results_sorted[1]]+': '+str(results[results_sorted[1]])+'\n'+LABELS[results_sorted[2]]+': '+str(results[results_sorted[2]])
    print(labels_sorted)
    return labels_sorted
