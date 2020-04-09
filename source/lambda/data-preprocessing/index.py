import os
import boto3
from time import strftime, gmtime

s3_client = boto3.client('s3')


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


def prepare_preprocessing_inputs(timestamp):
    print("Preparing Inputs")
    copy_source = {
        'Bucket': os.environ['processing_job_s3_bucket'],
        'Key': os.environ['processing_job_s3_raw_data_key']
    }
    key = os.path.join(os.environ['processing_job_input_s3_prefix'], timestamp)

    destination = get_full_s3_path(os.environ['processing_job_s3_bucket'], key)
    print("Copying raw data from {} to {}".format(get_full_s3_path(copy_source['Bucket'], copy_source['Key']),
                                                  destination))
    objects = s3_client.list_objects_v2(Bucket=copy_source['Bucket'], Prefix=copy_source['Key'])
    files = [content['Key'] for content in objects['Contents']]
    verify_files(files)

    for file in files:
        copy_source['Key'] = file
        dest = os.path.join(key, os.path.basename(file))
        s3_client.copy(copy_source, os.environ['processing_job_s3_bucket'], dest)

    return get_full_s3_path(os.environ['processing_job_s3_bucket'], key)


def prepare_preprocessing_output(event_source_s3, timestamp):
    print("Preparing Output")
    copy_source = {
        'Bucket': event_source_s3['bucket']['name'],
        'Key': event_source_s3['object']['key']
    }

    key = os.path.join(os.environ['processing_job_output_s3_prefix'],
                       timestamp,
                       os.path.basename(event_source_s3['object']['key']))

    destination = get_full_s3_path(os.environ['processing_job_s3_bucket'], key)
    print("Copying new accounts from {} to {}".format(get_full_path(event_source_s3), destination))
    s3_client.copy(copy_source, os.environ['processing_job_s3_bucket'], key)
    return get_full_s3_path(os.environ['processing_job_s3_bucket'],
                            os.path.join(os.environ['processing_job_output_s3_prefix'], timestamp))


def verify_files(files):
    expected_files = ['relations.csv.gz', 'usersdata.csv.gz']
    if not all([file in list(map(os.path.basename, files)) for file in expected_files]):
        raise Exception("Raw data absent or incomplete in {}".format(get_full_s3_path(
            os.environ['processing_job_s3_bucket'], os.environ['processing_job_s3_raw_data_key'])))


def get_full_s3_path(bucket, key):
    return os.path.join('s3://', bucket, key)


def get_full_path(event_source_s3):
    return get_full_s3_path(event_source_s3['bucket']['name'], event_source_s3['object']['key'])


def run_preprocessing_job(input, output, timestamp):
    print("Creating SageMaker Processing job with inputs from {} and outputs to {}".format(input, output))

    sagemaker_client = boto3.client('sagemaker')

    region = boto3.session.Session().region_name
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    ecr_repository_uri = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account_id,
                                                                        region,
                                                                        os.environ['processing_job_ecr_repository'])

    # upload code
    code_file = 'data-preprocessing/graph_data_preprocessor.py'
    code_file_s3_key = os.path.join(os.environ['processing_job_input_s3_prefix'], timestamp, code_file)
    s3_client.upload_file(code_file,
                          os.environ['processing_job_s3_bucket'],
                          code_file_s3_key)

    entrypoint = ["python3"] + [os.path.join("/opt/ml/processing/input/code",
                                             os.path.basename(code_file))]

    app_spec = {
        'ImageUri': ecr_repository_uri,
        'ContainerEntrypoint': entrypoint,
        # 'ContainerArguments': []
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
                'S3Uri': get_full_s3_path(os.environ['processing_job_s3_bucket'], code_file_s3_key),
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
            'InstanceType': os.environ['processing_job_instance_type'],
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
                                                      RoleArn=os.environ['processing_job_role_arn'])
    return response
