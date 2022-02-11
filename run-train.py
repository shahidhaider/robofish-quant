from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()


    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/Data/50'))



    experiment = Experiment(workspace=ws, name='robofish-ml')
    config = ScriptRunConfig(source_directory='./src',
                             script='train.py',
                             compute_target='shaider1',
                             arguments=[
                            '--data_path', dataset.as_named_input('input').as_mount(),
                            '--epochs', 50,
                            '--gpu',0,
                            '--recompule_ds',1]
                            )

    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='nn',
        file_path='./environment.yaml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)