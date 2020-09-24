#!/usr/bin/env bash

image=$1

region=$2

account=$3

# Get the region defined in the current configuration (default to us-east-1 if none defined)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --registry-ids ${account} --no-include-email)


# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t ${image} container
docker tag ${image} ${fullname}

docker push ${fullname}