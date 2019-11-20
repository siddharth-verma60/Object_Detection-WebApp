import tensorflow as tf
import tensornets as nets


def init():
	inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
	model = nets.YOLOv2(inputs, nets.Darknet19)
	trained_model = model.pretrained()

	return trained_model, model, inputs

