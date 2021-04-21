import oyaml as yaml

def generate_yamlfile(nr_of_clients=2, use_nvidia=True, add_network=True):

    clients = {}
    for i in range(nr_of_clients):

        client = {'environment': ['GET_HOSTS_FROM=dns'],
                  'image': 'keras-cifar-client:latest',
                  'build': {'context': '.'},
                  'working_dir': '/ app'}

        if use_nvidia:
            client['runtime'] = 'nvidia'
        client['command'] = '/bin/bash - c "fedn run client -in fedn-network.yaml"'
        client['volumes'] = ['./data/' + str(nr_of_clients) + 'clients/client' + str(i) + ':/app/data']
        clients['client'+str(i)] = client

    dict_file = {'version': '3.3', 'services': clients}

    if add_network:
        dict_file['networks'] = {'defaults': {'external': {'name': 'fedn_default'}}}

    with open(r'docker-compose-' + str(nr_of_clients) + 'clients.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)
