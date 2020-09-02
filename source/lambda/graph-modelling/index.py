import os
import boto3
import json
import tarfile
from time import strftime, gmtime

s3_client = boto3.client('s3')
S3_BUCKET = os.environ['training_job_s3_bucket']
OUTPUT_PREFIX = os.environ['training_job_output_s3_prefix']
INSTANCE_TYPE = os.environ['training_job_instance_type']
ROLE_ARN = os.environ['training_job_role_arn']


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
    minimum_expected_files = ['features.csv', 'tags.csv']

    if not all([file in [os.path.basename(s3_file) for s3_file in files] for file in minimum_expected_files]):
        return False, "Training data absent or incomplete in {}".format(full_s3_training_folder)

    return full_s3_training_folder, "Minimum files needed for training present in {}".format(full_s3_training_folder)


def run_modelling_job(timestamp,
                      train_input,
                      s3_bucket=S3_BUCKET,
                      train_out_prefix=OUTPUT_PREFIX,
                      train_job_prefix='sagemaker-graph-fraud-model-training',
                      train_source_dir='dgl_fraud_detection',
                      train_entry_point='train_dgl_mxnet_entry_point.py',
                      framework='mxnet',
                      framework_version='1.6.0',
                      xpu='gpu',
                      python_version='py3',
                      instance_type=INSTANCE_TYPE
                      ):
    print("Creating SageMaker Training job with inputs from {}".format(train_input))

    sagemaker_client = boto3.client('sagemaker')
    region = boto3.session.Session().region_name

    container = "763104351884.dkr.ecr.{}.amazonaws.com/{}-training:{}-{}-{}".format(region,
                                                                                    framework,
                                                                                    framework_version,
                                                                                    xpu,
                                                                                    python_version)

    training_job_name = "{}-{}".format(train_job_prefix, timestamp)

    code_path = tar_and_upload_to_s3(train_source_dir,
                                     s3_bucket,
                                     os.path.join(train_out_prefix, training_job_name, 'source'))

    framework_params = {
        'sagemaker_container_log_level': str(20),
        'sagemaker_enable_cloudwatch_metrics': 'false',
        'sagemaker_job_name': json.dumps(training_job_name),
        'sagemaker_program': json.dumps(train_entry_point),
        'sagemaker_region': json.dumps(region),
        'sagemaker_submit_directory': json.dumps(code_path)
    }

    model_params = {
          'nodes': 'features.csv',
          'edges': 'relation*',
          'labels': 'tags.csv',
          'model': 'rgcn',
          'num-gpus': 1,
          'batch-size': 10000,
          'embedding-size': 64,
          'n-neighbors': 1000,
          'n-layers': 2,
          'n-epochs': 10,
          'optimizer': 'adam',
          'lr': 1e-2
    }
    model_params = {k: json.dumps(str(v)) for k, v in model_params.items()}

    model_params.update(framework_params)

    train_params = \
        {
            'TrainingJobName': training_job_name,
            "AlgorithmSpecification": {
                "TrainingImage": container,
                "TrainingInputMode": "File"
            },
            "RoleArn": ROLE_ARN,
            "OutputDataConfig": {
                "S3OutputPath": get_full_s3_path(s3_bucket, train_out_prefix)
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
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


def tar_and_upload_to_s3(source, s3_bucket, s3_key):
    filename = "/tmp/sourcedir.tar.gz"
    with tarfile.open(filename, mode="w:gz") as t:
        for file in os.listdir(source):
            t.add(os.path.join(source, file), arcname=file)

    s3_client.upload_file(filename, s3_bucket, os.path.join(s3_key, 'sourcedir.tar.gz'))

    return get_full_s3_path(s3_bucket, os.path.join(s3_key, 'sourcedir.tar.gz'))


def get_full_s3_path(bucket, key):
    return os.path.join('s3://', bucket, key)


def get_full_path(event_source_s3):
    return get_full_s3_path(event_source_s3['bucket']['name'], event_source_s3['object']['key'])
