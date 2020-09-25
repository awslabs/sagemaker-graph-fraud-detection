import boto3
import sys

sys.path.append('./site-packages')
from crhelper import CfnResource

helper = CfnResource()


@helper.create
def on_create(_, __):
    pass

@helper.update
def on_update(_, __):
    pass


def delete_s3_objects(bucket_name):
    s3_resource = boto3.resource("s3")
    try:
        s3_resource.Bucket(bucket_name).objects.all().delete()
        print(
            "Successfully deleted objects in bucket "
            "called '{}'.".format(bucket_name)
        )
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print(
            "Could not find bucket called '{}'. "
            "Skipping delete.".format(bucket_name)
        )

def delete_ecr_images(repository_name):
    ecr_client = boto3.client("ecr")
    try:
        images = ecr_client.describe_images(repositoryName=repository_name)
        image_details = images["imageDetails"]
        if len(image_details) > 0:
            image_ids = [
                {"imageDigest": i["imageDigest"]} for i in image_details
            ]
            ecr_client.batch_delete_image(
                repositoryName=repository_name, imageIds=image_ids
            )
            print(
                "Successfully deleted {} images from repository "
                "called '{}'. ".format(len(image_details), repository_name)
            )
        else:
            print(
                "Could not find any images in repository "
                "called '{}' not found. "
                "Skipping delete.".format(repository_name)
            )
    except ecr_client.exceptions.RepositoryNotFoundException:
        print(
            "Could not find repository called '{}' not found. "
            "Skipping delete.".format(repository_name)
        )



def delete_s3_bucket(bucket_name):
    s3_resource = boto3.resource("s3")
    try:
        s3_resource.Bucket(bucket_name).delete()
        print(
            "Successfully deleted bucket "
            "called '{}'.".format(bucket_name)
        )
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print(
            "Could not find bucket called '{}'. "
            "Skipping delete.".format(bucket_name)
        )


@helper.delete
def on_delete(event, __):
    
    # delete ecr container repo
    repository_name = event["ResourceProperties"]["ECRRepository"]
    delete_ecr_images(repository_name)

    # remove files in s3 and delete bucket
    solution_bucket = event["ResourceProperties"]["SolutionS3BucketName"]
    delete_s3_objects(solution_bucket)
    delete_s3_bucket(solution_bucket)


def handler(event, context):
    helper(event, context)
