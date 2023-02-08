layer_nums = {
    'conv': 1,
    'pool': 1,
    'fc': 1,
    'softmax': 1
}

network_name = 'cnn'
models_path = 'models_weight'

# Path: utils/config.py

def init():
    global layer_nums, network_name, models_path

    layer_nums = {
        'conv': 1,
        'pool': 1,
        'fc': 1,
        'softmax': 1
    }

    network_name = 'cnn'
    models_path = 'models'

def get_network_name():
    if network_name == None:
        raise Exception('Network name is not defined')
    return network_name

def get_models_path():
    if models_path == None:
        raise Exception('Models path is not defined')
    return models_path

def get_layer_num(layer_type):
    global layer_nums
    if layer_type not in layer_nums:
        raise Exception('Layer type is not defined')
    return layer_nums[layer_type]


def set_network_name(name):
    global network_name
    network_name = name

def set_models_path(path):
    global models_path
    models_path = path

def set_layer_num(layer_type, num):
    global layer_nums
    if layer_type not in layer_nums:
        raise Exception('Layer type is not defined')
    layer_nums[layer_type] = num

def increment_layer_num(layer_type):
    global layer_nums
    if layer_type not in layer_nums:
        raise Exception('Layer type is not defined')
    layer_nums[layer_type] += 1