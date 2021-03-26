from keras_models.vgg import VGG
from fedn.utils.kerasweights import KerasWeightsHelper

import numpy as np

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = VGG(dimension='VGG11')
	outfile_name = "seed/VGG11_keras.npz"
	weights = model.get_weights()
	helper = KerasWeightsHelper()
	helper.save_model(weights, outfile_name)
	#np.savez_compressed(outfile_name, **model.state_dict())