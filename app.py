import base64
import os
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from load import init
import numpy as np
import tensorflow as tf

# initalize our flask app
app = Flask(__name__)

global weight_loaded_model, model, inputs
weight_loaded_model, model, inputs = init()

INPUT_HEIGHT = 416
INPUT_WIDTH = 416
INPUT_CHANNELS = 3


# decoding an image from base64 into raw representation
def convertImage(imgData):
    imgData = imgData.decode('ascii').split(',')[1].encode('ascii')
    with open('output.png', 'wb') as output:
        output.write(base64.decodebytes(imgData))


@app.route('/')
def index():
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


def get_score(boxes, class_index):
    max = 0
    for box in boxes[class_index]:
        if box[4] > max:
            max = box[4]

    return max


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()

    # encode it into a suitable format
    convertImage(imgData)
    x = load_img('output.png', target_size=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS))
    x = np.asarray(x, dtype=np.float32)
    imgs = np.array([x])
    imgs = imgs / 255

    with tf.Session() as sess:
        sess.run(weight_loaded_model)
        pred = sess.run(model, {inputs: imgs})
        boxes = model.get_boxes(pred, x.shape[1:3])

        if len(boxes[11]) > 0 and len(boxes[7]) > 0:
            dog_perc = int(get_score(boxes, 11) * 100)
            cat_perc = int(get_score(boxes, 7) * 100)
            response = "Both Dog ({}%) and Cat ({}%) present.".format(dog_perc, cat_perc)
        elif len(boxes[11] > 0):
            dog_perc = int(get_score(boxes, 11) * 100)
            response = "Only Dog ({}%) present.".format(dog_perc)
        elif len(boxes[7] > 0):
            cat_perc = int(get_score(boxes, 7) * 100)
            response = "Only Cat ({}%) present.".format(cat_perc)
        else:
            response = "Dog or Cat not found."

        return response


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=port)
