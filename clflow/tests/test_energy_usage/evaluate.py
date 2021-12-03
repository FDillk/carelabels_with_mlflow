import os
import pandas as pd

def evaluate(run_path):
    emissions = pd.read_csv(os.path.join(run_path, 'artifacts', 'emissions.csv'))
    return emissions.to_dict('records')[0]
