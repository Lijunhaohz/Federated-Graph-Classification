import yaml

from src.federated_methods import GC_Train


if __name__ == "__main__":
    model = 'GCFL+'
    dataset = 'PROTEINS'

    ######################## Load the Configuration ########################
    config_file = f"src/configs/config_gc_{model}.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    config['data_group'] = dataset

    ######################## Call the Library ########################
    assert model in ['SelfTrain', 'FedAvg', 'FedProx', 'GCFL', 'GCFL+', 'GCFL+dWs'], \
        f"Unknown model: {model}"
    GC_Train(config=config)