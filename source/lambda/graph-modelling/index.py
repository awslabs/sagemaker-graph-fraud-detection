import os
import boto3
import tarfile
from time import strftime, gmtime

s3_client = boto3.client('s3')


def process_event(event, context):
    print(event)

    event_source_s3 = event['Records'][0]['s3']

    print("S3 Put event source: {}".format(get_full_path(event_source_s3)))
    train_input, msg = verify_modelling_inputs(event_source_s3)
    print(msg)
    if not train_input:
        return msg
    timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    response = run_modelling_job(timestamp, train_input)
    return response


def verify_modelling_inputs(event_source_s3):
    training_kickoff_signal = "tags.csv"
    if training_kickoff_signal not in event_source_s3['object']['key']:
        msg = "Event source was not the training signal. Triggered by {} but expected folder to contain {}"
        return False, msg.format(get_full_s3_path(event_source_s3['bucket']['name'], event_source_s3['object']['key']),
                                 training_kickoff_signal)

    training_folder = os.path.dirname(event_source_s3['object']['key'])
    full_s3_training_folder = get_full_s3_path(event_source_s3['bucket']['name'], training_folder)

    objects = s3_client.list_objects_v2(Bucket=event_source_s3['bucket']['name'],  Prefix=training_folder)
    files = [content['Key'] for content in objects['Contents']]
    print("Contents of training data folder :")
    print("\n".join(files))
    minimum_expected_files = ['user_features.csv', 'tags.csv']

    if not all([file in [os.path.basename(s3_file) for s3_file in files] for file in minimum_expected_files]):
        return False, "Training data absent or incomplete in {}".format(full_s3_training_folder)

    return full_s3_training_folder, "Minimum files needed for training present in {}".format(full_s3_training_folder)


def run_modelling_job(timestamp, train_input):
    print("Creating SageMaker Training job with inputs from {}".format(train_input))

    sagemaker_client = boto3.client('sagemaker')
    region = boto3.session.Session().region_name

    container = "520713654638.dkr.ecr.{}.amazonaws.com/sagemaker-mxnet:1.4.1-gpu-py3".format(region)
    role = os.environ['training_job_role_arn']

    training_job_name = "sagemaker-graph-fraud-model-training-{}".format(timestamp)

    code_path = tar_and_upload_to_s3('dgl-fraud-detection', os.path.join(os.environ['training_job_output_s3_prefix'],
                                                                         training_job_name, 'source'))

    framework_params = {
        'sagemaker_container_log_level': str(20),
        'sagemaker_enable_cloudwatch_metrics': 'false',
        'sagemaker_job_name': training_job_name,
        'sagemaker_program': "train_dgl_entry_point.py",
        'sagemaker_region': region,
        'sagemaker_submit_directory': code_path
    }

    model_params = {
        'nodes': 'user_features.csv',
        'edges': get_edgelist(train_input),
        'labels': 'tags.csv',
        'model': 'rgcn',
        'num-gpus': str(1),
        'embedding-size': str(64),
        'n-layers': str(1),
        'n-epochs': str(100),
        'optimizer': 'adam',
        'lr': str(1e-2)
    }

    model_params.update(framework_params)

    train_params = \
        {
            'TrainingJobName': training_job_name,
            "AlgorithmSpecification": {
                "TrainingImage": container,
                "TrainingInputMode": "File"
            },
            "RoleArn": role,
            "OutputDataConfig": {
                "S3OutputPath": get_full_s3_path(os.environ['training_job_s3_bucket'],
                                                 os.path.join(os.environ['training_job_output_s3_prefix'],
                                                              training_job_name, 'output'))
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": os.environ['training_job_instance_type'],
                "VolumeSizeInGB": 30
            },
            "HyperParameters": model_params,
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 86400
            },
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": train_input,
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                },
            ]
        }

    response = sagemaker_client.create_training_job(**train_params)
    return response


def tar_and_upload_to_s3(source_file, s3_key):
    filename = "/tmp/source.tar.gz"
    with tarfile.open(filename, mode="w:gz") as t:
        t.add(source_file, arcname=os.path.basename(source_file))

    s3_client.upload_file(filename,
                          os.environ['training_job_s3_bucket'],
                          s3_key)

    return get_full_s3_path(os.environ['training_job_s3_bucket'], s3_key)


def get_edgelist(training_folder):
    training_folder = "/".join(training_folder.replace('s3://', '').split("/")[1:])
    objects = s3_client.list_objects_v2(Bucket=os.environ['training_job_s3_bucket'], Prefix=training_folder)
    files = [content['Key'] for content in objects['Contents']]
    bipartite_edges = ",".join(map(lambda x: x.split("/")[-1], [file for file in files if "relation" in file]))
    return bipartite_edges


def get_full_s3_path(bucket, key):
    return os.path.join('s3://', bucket, key)


def get_full_path(event_source_s3):
    return get_full_s3_path(event_source_s3['bucket']['name'], event_source_s3['object']['key'])

