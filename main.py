import argparse

from clflow.experiment_generation import generate_experiment, run_experiment, set_tracking_uri
from create_models import train_neural_network, train_random_forest


if __name__ == "__main__":

    # TODO missing: computing architektur & model version => use subdirectories or add MLModels hyperparameters?
    # TODO change torch data download to some central point

    parser = argparse.ArgumentParser(description='Measure energy usage when running the prediction')
    parser.add_argument('--data', type=str, choices=['MNIST', 'FashionMNIST', 'CIFAR10'], default='MNIST', help='data to use')
    parser.add_argument('--model', type=str, choices=['rf', 'dnn'], default='rf', help='model to use')
    parser.add_argument('--test', type=str, choices=['test_energy_usage'], default='test_energy_usage', help='model to use')
    args = parser.parse_args()

    # will train the model if not already trained
    if args.model == 'rf':
        train_random_forest(args.data)
    else:
        train_neural_network(args.data)

    # experiments are a single specific combination of data, model and test method
    set_tracking_uri('clruns')
    exp = generate_experiment(args.data, args.model, args.test)
    results = run_experiment(exp)

    # example output in form of command line
    test = results['meta']['test']
    data = results['meta']['entry_points']['main']['parameters']['data']['default']
    model = results['meta']['entry_points']['main']['parameters']['model']['default']
    duration = results['results']['duration'] # seconds
    en_consumtion = results['results']['energy_consumed'] # wattseconds
    print(f"Test {test} performed with {model} on {data} data:")
    print(f"Running took {duration:5.3f} seconds, with energy consumption of {en_consumtion*1e6:5.3f} µWs.")
