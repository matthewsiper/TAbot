import sys,os
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)
import argparse
import pika
import json
import datetime as dt

params = pika.ConnectionParameters(host='localhost',
                                   heartbeat=600,
                                   blocked_connection_timeout=300)
connection = pika.BlockingConnection(params)
read_channel = connection.channel()
read_channel.exchange_declare(exchange='uploader_exchange',
                              exchange_type='direct')

parser = argparse.ArgumentParser(description='Uploader process.')
parser.add_argument('--read_queue', type=str, default='batch')


if __name__ == '__main__':
    args = parser.parse_args()
    read_queue = args.read_queue
    read_channel.queue_declare(queue=read_queue, exclusive=False)
    read_channel.queue_bind(exchange='uploader_exchange', queue=read_queue, routing_key=read_queue)

    def callback(ch, method, properties, body):
        input_msg = json.loads(body)
        print(f'\nReceived ticker {input_msg["ticker"]} for pattern {input_msg["pattern"]}')
        command = f'aws s3 cp "/Users/mattscomputer/6513 P2/{input_msg["pattern"]}/{input_msg["pattern"]}_{input_msg["pattern_num"]}.jpg" s3://rb-dev-{input_msg["pattern"].replace("_","-")}'
        print(f"Command is {command}")
        os.system(command)
        print(f'\nUploaded {input_msg["ticker"]} to s3://rb-dev-{input_msg["pattern"].replace("_","-")}')

    read_channel.basic_consume(queue=read_queue, on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for tickers. To exit press CTRL+C')

    read_channel.start_consuming()

