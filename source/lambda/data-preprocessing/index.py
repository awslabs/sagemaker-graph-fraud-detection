import os
import boto3
import json
from time import strftime, gmtime

s3_client = boto3.client('s3')
S3_BUCKET = os.environ['processing_job_s3_bucket']
DATA_PREFIX = os.environ['processing_job_s3_raw_data_key']
INPUT_PREFIX = os.environ['processing_job_input_s3_prefix']
OUTPUT_PREFIX = os.environ['processing_job_output_s3_prefix']
INSTANCE_TYPE = os.environ['processing_job_instance_type']
ROLE_ARN = os.environ['processing_job_role_arn']
IMAGE_URI = os.environ['processing_job_ecr_repository']


def process_event(event, context):
    print(event)

    event_source_s3 = event['Records'][0]['s3']

    print("S3 Put event source: {}".format(get_full_path(event_source_s3)))
    timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    inputs = prepare_preprocessing_inputs(timestamp)
    outputs = prepare_preprocessing_output(event_source_s3, timestamp)
    response = run_preprocessing_job(inputs, outputs, timestamp)

    print(response)
    return response


def prepare_preprocessing_inputs(timestamp, s3_bucket=S3_BUCKET, data_prefix=DATA_PREFIX, input_prefix=INPUT_PREFIX):
    print("Preparing Inputs")
    key = os.path.join(input_prefix, timestamp)

    print("Copying raw data from {} to {}".format(get_full_s3_path(s3_bucket, data_prefix),
                                                  get_full_s3_path(s3_bucket, key)))
    objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=data_prefix)
    files = [content['Key'] for content in objects['Contents']]
    verify(files)

    for file in files:
        dest = os.path.join(key, os.path.basename(file))
        s3_client.copy({'Bucket': s3_bucket, 'Key': file}, s3_bucket, dest)

    return get_full_s3_path(s3_bucket, key)


def prepare_preprocessing_output(event_source_s3, timestamp, s3_bucket=S3_BUCKET, output_prefix=OUTPUT_PREFIX):
    print("Preparing Output")
    copy_source = {
        'Bucket': event_source_s3['bucket']['name'],
        'Key': event_source_s3['object']['key']
    }

    key = os.path.join(output_prefix, timestamp, os.path.basename(event_source_s3['object']['key']))

    destination = get_full_s3_path(s3_bucket, key)
    print("Copying new accounts from {} to {}".format(get_full_path(event_source_s3), destination))
    s3_client.copy(copy_source, s3_bucket, key)
    return get_full_s3_path(s3_bucket, os.path.join(output_prefix, timestamp))


def verify(files, s3_bucket=S3_BUCKET, data_prefix=DATA_PREFIX, expected_files=['transaction.csv', 'identity.csv']):
    if not all([file in list(map(os.path.basename, files)) for file in expected_files]):
        raise Exception("Raw data absent or incomplete in {}".format(get_full_s3_path(
            s3_bucket, data_prefix)))


def get_full_s3_path(bucket, key):
    return os.path.join('s3://', bucket, key)


def get_full_path(event_source_s3):
    return get_full_s3_path(event_source_s3['bucket']['name'], event_source_s3['object']['key'])


def run_preprocessing_job(input,
                          output,
                          timestamp,
                          s3_bucket=S3_BUCKET,
                          input_prefix=INPUT_PREFIX,
                          instance_type=INSTANCE_TYPE,
                          image_uri=IMAGE_URI
                          ):
    print("Creating SageMaker Processing job with inputs from {} and outputs to {}".format(input, output))

    sagemaker_client = boto3.client('sagemaker')

    region = boto3.session.Session().region_name
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    ecr_repository_uri = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account_id, region, image_uri)

    # upload code
    code_file = 'data-preprocessing/graph_data_preprocessor.py'
    code_file_s3_key = os.path.join(input_prefix, timestamp, code_file)
    s3_client.upload_file(code_file, s3_bucket, code_file_s3_key)

    entrypoint = ["python3"] + [os.path.join("/opt/ml/processing/input/code",
                                             os.path.basename(code_file))]

    app_spec = {
        'ImageUri': ecr_repository_uri,
        'ContainerEntrypoint': entrypoint,
        'ContainerArguments': ['--id-cols', 'card1,card2,card3,card4,card5,card6,ProductCD,addr1,addr2,P_emaildomain,R_emaildomain',
                                '--cat-cols','M1,M2,M3,M4,M5,M6,M7,M8,M9']
    }

    processing_inputs = [
        {
            'InputName': 'input1',
            'S3Input': {
                'S3Uri': input,
                'LocalPath': '/opt/ml/processing/input',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
            }
        },
        {
            'InputName': 'code',
            'S3Input': {
                'S3Uri': get_full_s3_path(s3_bucket, code_file_s3_key),
                'LocalPath': '/opt/ml/processing/input/code',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
            }
        },

    ]
    processing_output = {'Outputs': [{'OutputName': 'output1',
                                      'S3Output': {'S3Uri': output,
                                                   'LocalPath': '/opt/ml/processing/output',
                                                   'S3UploadMode': 'EndOfJob'}
                                      }]}

    processing_job_name = "sagemaker-graph-fraud-data-processing-{}".format(timestamp)
    resources = {
        'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': instance_type,
            'VolumeSizeInGB': 30
        }
    }

    network_config = {'EnableNetworkIsolation': False}
    stopping_condition = {'MaxRuntimeInSeconds': 3600}

    response = sagemaker_client.create_processing_job(ProcessingInputs=processing_inputs,
                                                      ProcessingOutputConfig=processing_output,
                                                      ProcessingJobName=processing_job_name,
                                                      ProcessingResources=resources,
                                                      StoppingCondition=stopping_condition,
                                                      AppSpecification=app_spec,
                                                      NetworkConfig=network_config,
                                                      RoleArn=ROLE_ARN)
    return response
