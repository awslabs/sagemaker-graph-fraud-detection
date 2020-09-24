import boto3
import sys
import time
from .logs import logs_for_build

def build(project_name, session=boto3.session.Session(), log=True):
    print("Starting a build job for CodeBuild project: {}".format(project_name))
    id = _start_build(session, project_name)
    if log:
        logs_for_build(id, wait=True, session=session)
    else:
        _wait_for_build(id, session)

def _start_build(session, project_name):
    args = {"projectName": project_name}
    client = session.client("codebuild")

    response = client.start_build(**args)
    return response["build"]["id"]


def _wait_for_build(build_id, session, poll_seconds=10):
    client = session.client("codebuild")
    status = client.batch_get_builds(ids=[build_id])
    while status["builds"][0]["buildStatus"] == "IN_PROGRESS":
        print(".", end="")
        sys.stdout.flush()
        time.sleep(poll_seconds)
        status = client.batch_get_builds(ids=[build_id])
    print()
    print(f"Build complete, status = {status['builds'][0]['buildStatus']}")
    print(f"Logs at {status['builds'][0]['logs']['deepLink']}")