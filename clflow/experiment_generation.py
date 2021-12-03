import os
import re
from shutil import copy, copytree, rmtree
from urllib.parse import urlparse, urljoin
from urllib.request import pathname2url
import importlib
import yaml

from mlflow.projects import run
import mlflow


# THE SPECIFIC, EXECUTABLE TESTS ARE REPRESENTED AS MLFLOW EXPERIMENTS!


def set_tracking_uri(local_dir):
    uri = urljoin('file:', pathname2url(os.path.abspath(local_dir)))
    mlflow.tracking.set_tracking_uri(uri)


def merge_requirements(requirements_lists, version_re='(\w*)\s*([=<>]*)\s*(.*)'):
    """merges several requirements"""
    final_requirements = {}
    for requirements in requirements_lists:
        req = [re.match(version_re, line).groups() for line in requirements if '#' not in line]
        for module, equality, version in req:
            # TODO improve merging of conflicting requirements
            if module in final_requirements and final_requirements[module][1] > version:
                print(f'Requirements version conflict for {module}, running test with {final_requirements[module][1]} instead of {version}!')
                continue
            final_requirements[module] = [equality, version]
    return [ f'{module}{equality}{version}' for module, (equality, version) in final_requirements.items() ]
            

def load_requirements(path, fname='requirements.txt'):
    with open(os.path.join(path, fname), 'r') as reqf:
        lines = reqf.read().splitlines()
    return lines


def load_yaml(path, fname='conda.yaml'):
    with open(os.path.join(path, fname), 'r') as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data


def create_environment(data, model, test, env_out):
    """create a new experiment environment file"""
    # read all requirements
    model = os.path.join('clflow', 'models', model)
    data = os.path.join('clflow', 'data', data)
    model_yaml = load_yaml(model)
    data_req = load_requirements(data)
    test_req = load_requirements(test)
    # merge into MLproject yaml
    model_yaml['dependencies'][2]['pip'] = merge_requirements([model_yaml['dependencies'][2]['pip'], data_req, test_req])
    with open(env_out, 'w') as yaml_output:
        yaml.dump(model_yaml, yaml_output, default_flow_style=False, explicit_start=True, allow_unicode=True)


def generate_experiment(data, model, test, exp_root='clexperiments', env='environment.yml', override=False):
    """generates a new experiment that can then be executed"""
    model = f'{model}_{data}' # for now, final model name is composed of test and model name
    # create experiment directory containing the MLproject
    exp_name = f'{test}_{model}'
    exp_path = os.path.join(exp_root, exp_name)
    if override and os.path.exists(exp_path):
        # delete directory and entry from mlflow database
        exp = mlflow.get_experiment_by_name(exp_name)
        mlflow.delete_experiment(exp.experiment_id)
        rmtree(exp_path)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        # add relative clflow paths
        test = os.path.join('clflow', 'tests', test)
        # load entry_points for test and fill in data and model arguments
        entry_points = load_yaml(test, 'entry_points.yaml')
        for entry in entry_points.values():
            entry['parameters']['model']['default'] = model
            entry['parameters']['data']['default'] = data
        # create ML project file
        mlproject = { 'name': test, 'conda_env': env, 'entry_points' : entry_points }
        with open(os.path.join(exp_path, "MLproject"), 'w') as mlproj_file:
            yaml.dump(mlproject, mlproj_file, default_flow_style=False, explicit_start=True, allow_unicode=True)
        # create environment file, copy model data and test code
        create_environment(data, model, test, os.path.join(exp_path, env))
        copy(os.path.join(test, 'run.py'), exp_path)
        copy(os.path.join(test, 'evaluate.py'), exp_path)
        copy(os.path.join('clflow', 'data', 'loader.py'), exp_path)
        copytree(os.path.join('clflow', 'models', model), os.path.join(exp_path, model))
        copytree(os.path.join('clflow', 'data', data), os.path.join(exp_path, data))
        mlflow.create_experiment(exp_name) # add the experiment to MLflow database
    return exp_path


def evaluate_meta(run_path):
    """evaluates generic meta information that every test needs to contain"""

    exp_name = load_yaml(os.path.dirname(run_path), 'meta.yaml')['name']
    run_meta = load_yaml(run_path, 'meta.yaml')
    run_meta['experiment_name'] = exp_name

    # remove redundant keys
    for key in ['name', 'lifecycle_stage', 'run_uuid', 'source_name', 'source_type', 'source_version', 'status']:
        del run_meta[key]

    # store additional information from MLproject file
    mlproject = load_yaml(urlparse(run_meta['artifact_uri']).path, 'MLproject')
    run_meta['test'] = mlproject['name']
    run_meta['entry_points'] = mlproject['entry_points']
    
    return run_meta


def run_experiment(exp_path):
    """runs a new run of the given experiment"""
    # get underlying experiment and tracking information
    tracking_path = urlparse(mlflow.tracking.get_tracking_uri()).path
    exp_name = os.path.basename(exp_path)
    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    # start run
    run_id = mlflow.start_run(experiment_id=exp_id).info.run_id
    test_run = run(exp_path, experiment_id=exp_id, run_id=run_id)
    mlflow.log_artifact(os.path.join(exp_path, 'MLproject'))
    mlflow.end_run()
    evaluate_results = importlib.import_module(exp_path.replace('/', '.') + '.evaluate').evaluate
    # evaluate
    run_path = os.path.join(tracking_path, exp_id, test_run.run_id)
    results = {
        'meta': evaluate_meta(run_path),
        'results': evaluate_results(run_path)
    }
    return results
