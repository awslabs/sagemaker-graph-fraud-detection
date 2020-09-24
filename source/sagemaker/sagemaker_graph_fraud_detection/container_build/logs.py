import collections
import sys
import time

import botocore


# Log tailing code taken as-is from sagemaker_run_notebook.
# https://github.com/aws-samples/sagemaker-run-notebook/blob/master/sagemaker_run_notebook/container_build.py#L100
# It can be added as a dependency once its on PyPI.


class LogState(object):
    STARTING = 1
    WAIT_IN_PROGRESS = 2
    TAILING = 3
    JOB_COMPLETE = 4
    COMPLETE = 5


# Position is a tuple that includes the last read timestamp and the number of items that were read
# at that time. This is used to figure out which event to start with on the next read.
Position = collections.namedtuple("Position", ["timestamp", "skip"])


def log_stream(client, log_group, stream_name, position):
    """A generator for log items in a single stream. This will yield all the
    items that are available at the current moment.

    Args:
        client (boto3.CloudWatchLogs.Client): The Boto client for CloudWatch logs.
        log_group (str): The name of the log group.
        stream_name (str): The name of the specific stream.
        position (Position): A tuple with the time stamp value to start reading the logs from and
                             The number of log entries to skip at the start. This is for when
                             there are multiple entries at the same timestamp.
        start_time (int): The time stamp value to start reading the logs from (default: 0).
        skip (int): The number of log entries to skip at the start (default: 0). This is for when there are
                    multiple entries at the same timestamp.

    Yields:
        A tuple with:
        dict: A CloudWatch log event with the following key-value pairs:
             'timestamp' (int): The time of the event.
             'message' (str): The log event data.
             'ingestionTime' (int): The time the event was ingested.
        Position: The new position
    """

    start_time, skip = position
    next_token = None

    event_count = 1
    while event_count > 0:
        if next_token is not None:
            token_arg = {"nextToken": next_token}
        else:
            token_arg = {}

        response = client.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            startTime=start_time,
            startFromHead=True,
            **token_arg,
        )
        next_token = response["nextForwardToken"]
        events = response["events"]
        event_count = len(events)
        if event_count > skip:
            events = events[skip:]
            skip = 0
        else:
            skip = skip - event_count
            events = []
        for ev in events:
            ts, count = position
            if ev["timestamp"] == ts:
                position = Position(timestamp=ts, skip=count + 1)
            else:
                position = Position(timestamp=ev["timestamp"], skip=1)
            yield ev, position


# Copy/paste/slight mods from session.logs_for_job() in the SageMaker Python SDK
def logs_for_build(
    build_id, session, wait=False, poll=10
):  # noqa: C901 - suppress complexity warning for this method
    """Display the logs for a given build, optionally tailing them until the
    build is complete.

    Args:
        build_id (str): The ID of the build to display the logs for.
        wait (bool): Whether to keep looking for new log entries until the build completes (default: False).
        poll (int): The interval in seconds between polling for new log entries and build completion (default: 10).
        session (boto3.session.Session): A boto3 session to use

    Raises:
        ValueError: If waiting and the build fails.
    """

    codebuild = session.client("codebuild")
    description = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
    status = description["buildStatus"]

    log_group = description["logs"].get("groupName")
    stream_name = description["logs"].get("streamName")  # The list of log streams
    position = Position(
        timestamp=0, skip=0
    )  # The current position in each stream, map of stream name -> position

    # Increase retries allowed (from default of 4), as we don't want waiting for a build
    # to be interrupted by a transient exception.
    config = botocore.config.Config(retries={"max_attempts": 15})
    client = session.client("logs", config=config)

    job_already_completed = False if status == "IN_PROGRESS" else True

    state = (
        LogState.STARTING if wait and not job_already_completed else LogState.COMPLETE
    )
    dot = True

    while state == LogState.STARTING and log_group == None:
        time.sleep(poll)
        description = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
        log_group = description["logs"].get("groupName")
        stream_name = description["logs"].get("streamName")

    if state == LogState.STARTING:
        state = LogState.TAILING

    # The loop below implements a state machine that alternates between checking the build status and
    # reading whatever is available in the logs at this point. Note, that if we were called with
    # wait == False, we never check the job status.
    #
    # If wait == TRUE and job is not completed, the initial state is STARTING
    # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is complete).
    #
    # The state table:
    #
    # STATE               ACTIONS                        CONDITION               NEW STATE
    # ----------------    ----------------               -------------------     ----------------
    # STARTING            Pause, Get Status              Valid LogStream Arn     TAILING
    #                                                    Else                    STARTING
    # TAILING             Read logs, Pause, Get status   Job complete            JOB_COMPLETE
    #                                                    Else                    TAILING
    # JOB_COMPLETE        Read logs, Pause               Any                     COMPLETE
    # COMPLETE            Read logs, Exit                                        N/A
    #
    # Notes:
    # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to Cloudwatch after
    #   the build was marked complete.
    last_describe_job_call = time.time()
    dot_printed = False
    while True:
        for event, position in log_stream(client, log_group, stream_name, position):
            print(event["message"].rstrip())
            if dot:
                dot = False
                if dot_printed:
                    print()
        if state == LogState.COMPLETE:
            break

        time.sleep(poll)
        if dot:
            print(".", end="")
            sys.stdout.flush()
            dot_printed = True
        if state == LogState.JOB_COMPLETE:
            state = LogState.COMPLETE
        elif time.time() - last_describe_job_call >= 30:
            description = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
            status = description["buildStatus"]

            last_describe_job_call = time.time()

            status = description["buildStatus"]

            if status != "IN_PROGRESS":
                print()
                state = LogState.JOB_COMPLETE

    if wait:
        if dot:
            print()