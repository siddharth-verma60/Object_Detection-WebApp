# How to run the webapp


## Overview

This is the code for detecting cat and dog in the image. It uses an object detection algorithm (YOLO).

## Description

Here, our task is to detect whether a cat or dog is present in an image or not.

There are mainly four case to cover:
 1.) Both cat and dog are present.
 2.) Only cat is present.
 3.) Only dog is present.
 4.) None of the animal is present.
 
 We can use a classification or object detection model for this use case. This code employs an
 object detection algorithm, YOLO (You only look once). We use a pretrained network based on [Darknet](https://github.com/pjreddie/darknet)
 implementation, provided by tensornets library. The network is trained on [VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
 dataset which contains images of common objects and things.
 We preprocess the images and feed into our network. The network outputs bounding boxes around
 the animals, which we process and send only the desired output to the webpage.

## Dependencies

```sudo pip install -r requirements.txt```

## Usage

Once dependencies are installed, just run this to see it in your browser. 

```python app.py```

That's it! It's serving a saved model from tensornets library via Flask on port 5000 of your localhost. 


