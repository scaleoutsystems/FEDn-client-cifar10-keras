import numpy as np
import os
from client.models.keras_models.vgg import VGG
from fedn.utils.kerashelper import KerasHelper


def generate_seed_model(model_dimension='VGG11'):

	model = VGG(dimension=model_dimension)
	outfile_name = 'seed/' + model_dimension + '_keras.npz'
	weights = model.get_weights()
	helper = KerasHelper()

	if not os.path.exists('seed'):
		os.makedirs('seed')
	helper.save_model(weights, outfile_name)


if __name__ == '__main__':

	with open('client/settings.yaml', 'r') as fh:
		try:
			settings = dict(yaml.safe_load(fh))
		except yaml.YAMLError as e:
			raise (e)

	generate_seed_model(model_dimension=settings['model_dimension'])


