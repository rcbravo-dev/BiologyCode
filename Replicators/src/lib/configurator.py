import numpy as np
npr = np.random
from dataclasses import dataclass
from typing import Any
import getpass

@dataclass
class Configurator:
    params: dict

    def __post_init__(self):
        # Iterate through params and set them to their default values
        for param, value in self.params.items():
            setattr(self, param, value)

        # After all params are set, lock the object and delete the params attribute
        self.locked = True
        del self.params

    def __setattr__(self, name: str, value: Any) -> None:
        # If the object is locked, do not allow any changes
        if self._assignment_locked():
            pass
        # If not locked, check if the parameter is valid
        else:
            try:
                if name in {'params', 'locked'}:
                    pass
                else:
                    self._validate_param(name, value)
            except AssertionError as error:
                # When not in range or option set
                raise error
            except Exception as error:
                # Invalaid parameter
                pass
            else:
                super().__setattr__(name, value)
        
    def _assignment_locked(self):
        '''Check if the object is locked. If it is, return True. 
        Otherwise, return False. If the attribute "locked" is not
        defined, set it to False.'''
        locked = False

        try:
            if self.locked:
                locked = True
        except AttributeError:
            super().__setattr__('locked', False)
        finally:
            return locked

    def _validate_param(self, name: str, value: Any):
        '''I want the user to be able to change params but not to set them to invalid values. 
        This function checks if the value is valid. __post_init__ 
        deletes the params attribute, so this function will raise 
        an AttributeError after the lock is set.'''
        
        if name in self.params:
            # If there is an option or range, check if the value is valid
            if f'{name}_option' in self.params:
                opt = self.params[f'{name}_option']
                assert value in opt, f"Invalid option for {name}: {value}. Expected option in set: {opt}"
        
            elif f'{name}_range' in self.params:
                rng = self.params[f'{name}_range']
                assert rng[0] <= value <= rng[1], f"Invalid range for {name}: {value}. Expected range: {rng}"
            
            # Don't store the value if it is a range or option set
            elif name.endswith('_range') or name.endswith('_option'):
                raise ValueError(f"Invalid parameter: {name}")
            
        else:
            raise ValueError(f"Invalid parameter: {name}")


def configuration_test(config: Configurator, verbose: bool = False):
    try:
        # Test that lock works
        config.locked = False
        assert config.locked == True

        # Test that assignment is locked
        config.secret_size = 4
        assert config.secret_size == 32
    except AssertionError:
        print("Assignment is not locked")
        return False
    else:
        print("\n\n Assignment test passed.")
        return True
    finally:
        if verbose:
            print()
            for key, value in config.__dict__.items():
                print(key, value)

def get_dtype_min_mid_max(dtype):
    max = np.iinfo(dtype).max 
    assert np.iinfo(dtype).min == 0
    return 0, (max + 1) // 2, max


GROUP_NAME = 'BiologyCode'
APPLICATION_NAME = 'Replicators'

# Environment
ENVIRONMENT = 'development'
USER = getpass.getuser()

INSTALL_PATH = f'/Users/Shared/SharedProjects/Projects/{GROUP_NAME}/{APPLICATION_NAME}'

# Logging
LOG_LEVEL = 'DEBUG'
LOG_STACK = False
LOG_EXC = False

# Networking
API_HOST = 'localhost'
API_PORT = 9999
HANDLER_TIMEOUT = 10.0

# Encryption
BITS = 8
DTYPE = np.uint8
SEED = 1234

PARAMS = {
    # Paths
    'cwd': f'{INSTALL_PATH}',
    'trash': f'/Users/{USER}/.Trash/',
    'data_path': f'{INSTALL_PATH}/simulation',

    # Environment
    'environment': ENVIRONMENT,
    'environment_option': {'development', 'alpha', 'beta', 'production'},

    # Logging
    'log_level': LOG_LEVEL,
    'log_level_option': {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
    'log_stack': LOG_STACK,
    'log_exc': LOG_EXC,

    # Objects
    'rng': npr.default_rng(SEED),

    # Networking
    'api_host': API_HOST,
    'api_port': API_PORT,
    'chunk_size': 1024,
    'handler_timeout': HANDLER_TIMEOUT,

    # Logging
    # 'https://stackoverflow.com/questions/7507825/where-is-a-complete-example-of-logging-config-dictconfig#7507842'
    'logging_config': { 
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': { 
            'standard': { 
                'format': '[%(levelname)s] %(asctime)s, %(name)s : %(message)s',
                'datefmt': '%j %H:%M:%S',  # Julian day, 24-hour clock
            },
        },
        'handlers': { 
            'console': { 
                'level': 'WARNING',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                # Default is stderr
                'stream': 'ext://sys.stdout',  
            },
            'clientFile': {
                'level': LOG_LEVEL,
                'formatter': 'standard',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': f'{INSTALL_PATH}/logs/operator_log.log',
                'mode': 'w',
                'maxBytes': 1048576,
                'backupCount': 3,
            },
            'serverFile': {
                'level': LOG_LEVEL,
                'formatter': 'standard',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': f'{INSTALL_PATH}/logs/server_log.log',
                # 'a' for append, 'w' for over-write
                'mode': 'w',
                'maxBytes': 1048576,
                'backupCount': 3,
            },
        },
        'loggers': { 
            # root logger
            '': {  
                'handlers': ['console', 'clientFile', 'serverFile'],
                'level': 'WARNING',
                'propagate': True
            },
            'CLIENT': { 
                'handlers': ['clientFile'],
                'level': LOG_LEVEL,
                'propagate': False
            },
            'SERVER': { 
                'handlers': ['serverFile'],
                'level': LOG_LEVEL,
                'propagate': False
            },
            'NET': { 
                'handlers': ['clientFile'],
                'level': LOG_LEVEL,
                'propagate': False
            },
            # if __name__ == '__main__'
            '__main__': {  
                'handlers': ['console'],
                'level': LOG_LEVEL,
                'propagate': False
            },
        }, 
    },
}

# Import this configuration object
configurations = Configurator(PARAMS)