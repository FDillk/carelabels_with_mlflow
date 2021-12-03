import os
import argparse

from codecarbon import EmissionsTracker

from mlflow.pyfunc import load_model
import mlflow

from loader import load_as_arrays # will succeed in final experiment



class MLFlowEmissionsTracker(EmissionsTracker):
    """EmissionsTracker that logs the emissions file output as MLflow artifact"""

    def stop(self):
        super().stop()
        emissions_fname = 'emissions.csv'
        mlflow.log_artifact(emissions_fname)
        os.remove(emissions_fname)


if __name__ == "__main__":

    mlflow.log_param('trackinguri', mlflow.tracking.get_tracking_uri())

    parser = argparse.ArgumentParser(description='Measure energy usage when running the prediction')
    parser.add_argument('model', type=str, help='path to model')
    parser.add_argument('data', type=str, help='path to data')
    args = parser.parse_args()

    # load model and data
    model = load_model(args.model)
    _, _, X_test, _ = load_as_arrays(args.data)

    # start tracking and run prediction
    tracker = MLFlowEmissionsTracker()
    tracker.start()
    model.predict(X_test)
    tracker.stop()
