from load_dataset import load_dataset
from generate_yamlfile import generate_yamlfile
from init_model import generate_seed_model
import sys
import yaml
if __name__ == '__main__':

    with open('client/settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise (e)
    nr_of_clients = int(sys.argv[1])
    # Load and partition the dataset
    load_dataset(path='data', nr_of_splits=nr_of_clients)
    # Generate docker-compose.yaml
    use_gpu = settings['device'] == 'cuda'
    generate_yamlfile(nr_of_clients=nr_of_clients, use_nvidia=use_gpu, add_network=True)
    # Generate seed model


    generate_seed_model(model_dimension=settings['model_dimension'])


