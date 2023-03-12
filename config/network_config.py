import global_config

class NetworkConfig():
    _sharedInstance = None

    @staticmethod
    def initialize(yaml_data, hyperparam_data):
        if(NetworkConfig._sharedInstance == None):
            NetworkConfig._sharedInstance = NetworkConfig(yaml_data, hyperparam_data)

    @staticmethod
    def getInstance():
        return NetworkConfig._sharedInstance

    def __init__(self, yaml_data, hyperparam_data):
        self.yaml_config = yaml_data
        self.hyperparam_config = hyperparam_data

    def get_network_config(self):
        return self.yaml_config

    def get_hyper_params(self):
        return self.hyperparam_config

    def get_version_name(self):
        general_config = global_config.general_config
        network_version = general_config["network_version"]
        iteration = general_config["iteration"]

        return str(network_version) + "_" + str(iteration)