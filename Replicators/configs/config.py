import numpy as np
import hashlib
import logging
import logging.config
import yaml

PROJECT_NAME = 'ZKP'
CWD = '/Users/Shared/SharedProjects/Projects/ZKP/ZKP/'
TRASH = '/Users/code/.Trash/'
LOG_LEVEL = 'DEBUG'
SEED = 1234
RNG = np.random.default_rng(seed=SEED)


def setup_logging(default_path=CWD + 'configs/logging_config.yaml'):
    with open(default_path, 'rt') as file:
        config = yaml.safe_load(file.read())
    logging.config.dictConfig(config)

    print(f'Logging configured in {PROJECT_NAME}')


configs = {
	'zkp':{
		'args':[],
		'kwargs':{
            # The size of the secret array. 
			# assert array_size % 8 == 0
			'array_size':32, 
            # The number of bits each element in the secret array will represent.
			# Options are: 8, 16, 32, 64 
			'bits':8,
            # The hashing protocol used.
			# Options are: 'sha256', 'sha512', 'blake2s', 'blake2b'
			'hash_protocol_string':'sha256', 
            # The maximum number of rotations performed IOT randomize the secret array.
			'max_rotations':30,
            'name_size':16,
		}
	},
}


