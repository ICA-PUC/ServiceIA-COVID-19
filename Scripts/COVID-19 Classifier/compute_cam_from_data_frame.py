
'''
- load dataset to be evaluated
- load trained model
- load function to compute CAM
- Save images in a folder

'''

# define functions

# import the necessary packages
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2



def build_pretrained_model(model_file, isTrain=True):
    with open(model_file + '.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_file + '_weights.best.hdf5')
    if isTrain is False:
        model.trainable = False
    return model



class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName

		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name

		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

	def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output, 
				self.model.output])

		# record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]

		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)

		# compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads

		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		# compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		# grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))

		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")

		# return the resulting heatmap to the calling function
		return heatmap

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)
    
#############################################################################


""" LOAD DATASET """

root_path = '/scratch/parceirosbr/bigoilict/share/GeoFacies/jose/.datasets/'

test_dataset = 'test_df_covid_softmax_offset'
test_df = pd.read_csv(test_dataset)

model_file = 'densenet121_covid19_softmax_weights_offset'

#############################################################################
# define image size
IMG_SIZE = (224, 224)
columns = ['normal', 'pneumonia', 'covid']

load_datagen = ImageDataGenerator()
load_test_data = load_datagen.flow_from_dataframe(dataframe=test_df,
                                    x_col='Image Path',
                                    y_col=columns,
                                    drop_duplicates=False,
                                    class_mode="raw",
                                    target_size=IMG_SIZE,
                                    color_mode = 'rgb',
                                    batch_size=len(test_df))
                                    # batch_size=64)

test_X, test_Y = next(load_test_data) # one big batch
test_X = preprocess_input(test_X)

model = build_pretrained_model(model_file, isTrain=False)

preds = model.predict(test_X)
# compute global metrics
preds = np.argmax(preds, axis=-1)
y_test = np.argmax(test_Y, axis=-1)


print('precision', precision_score(y_test, preds, average=None))
print('recall', recall_score(y_test, preds, average=None))
print('f1_score', f1_score(y_test, preds, average=None))

print('CM', confusion_matrix(y_test, preds))

images_dir = 'images/'

for index in range(len(test_Y)):
    
    image = test_X[index][np.newaxis]
    org = image[0]
    org = np.uint8(255*(org-org.min())/(org.max()-org.min()))
    # org = np.tile(image[0]*255, [1,1,3]).astype('uint8') # to rgb
    preds = model.predict(image)
    i = np.argmax(preds[0])
    # print('reference,', test_Y[index], 'prediction', np.around(preds[0], decimals=2), preds[0].sum())
    # print('reference,', test_Y[index], 'prediction', np.around(preds[0], decimals=2), preds[0].sum())
    # print(columns)
    # print('image', index)
    
    image_output_filename = images_dir +  columns[np.argmax(test_Y[index])] + '_' + str(index) + '.jpg'
    
    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (org.shape[1], org.shape[0]))
    
    (heatmap, output) = cam.overlay_heatmap(heatmap, org, alpha=0.7)
    output = np.hstack([org, heatmap, output])
    output = cv2.resize(output, (org.shape[1]*3, org.shape[0]))
    cv2.imwrite(image_output_filename, output)


