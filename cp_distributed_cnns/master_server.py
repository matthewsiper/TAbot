import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt

import json

import os
import boto3
from botocore.config import Config

import argparse
import pika
import json

conn = boto3.client(
    's3',
    aws_access_key_id='AKIAIWPOHHSFY2F4UEOA',
    aws_secret_access_key='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'
)

parser = argparse.ArgumentParser(description='Master node.')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=8220)

buckets = ["rb-dev-ta-pattern-horrizontal2", "rb-dev-ta-pattern-wedgedown2", "rb-dev-ta-pattern-wedgeup2"]
BASE_DIR_DATA = "/home/ec2-user/data/data{}"
BASE_DIR_MODELS = "/home/ec2-user/models/model{}"
BASE_DIR_PRED = "/home/ec2-user/data/preds/{}"


def bucket_to_dir(bucket_name):
    return '_'.join(bucket_name.split('-')[2:])

def load_image(bucket_name):
    file_name = None
    for key in conn.list_objects(Bucket=bucket_name)['Contents'][0]:
        file_name = key['Key']
        target = BASE_DIR_PRED.format(file_name)
        conn.download_file(bucket_name, file_name, target)
    return target



def issue_train():
    for q in ["queue1", "queue2", "queue3"]:
        channel.basic_publish(exchange='cnn_exchange',
                      routing_key=q,
                      body=json.dumps({"message": "train"}))
    return f"train command issued"


if __name__ == "__main__":
    args = parser.parse_args()
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange='cnn_exchange',
                             exchange_type='direct')

    for queue in ["queue1", "queue2", "queue3"]:
        channel.queue_declare(queue=queue, exclusive=False)
        channel.queue_bind(
            exchange='cnn_exchange', queue=queue, routing_key=queue)
