import sys,os
BASE_DIR = os.path.join(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)

from chart_builder_worker import ChartBuilderWorker


import argparse
import pika
import json

cb = ChartBuilderWorker()

params = pika.ConnectionParameters(host='localhost',
                                   heartbeat=600,
                                   blocked_connection_timeout=300)
# read queue
connection = pika.BlockingConnection(params)
read_channel = connection.channel()
read_channel.exchange_declare(exchange='cb_proc_exchange', exchange_type='direct')

# write queue
write_channel = connection.channel()

parser = argparse.ArgumentParser(description='CBWorker process.')
parser.add_argument('--read_queue', type=str, default='cb_queue1')
parser.add_argument('--write_queue', type=str, default='cb_queue_1')


if __name__ == '__main__':
    args = parser.parse_args()
    read_queue = args.read_queue
    write_queue = args.write_queue


    def callback(ch, method, properties, body):
        input_msg = json.loads(body)
        print(f"\nReceived input message: {input_msg}")
        hist_df = cb.get_price_data(input_msg["ticker"])
        cb.build_chart(input_msg["pattern"], hist_df, file_num=input_msg["pattern_num"])
        output_msg = json.dumps({"chart_path": f"/Users/mattscomputer/6513 P2/{input_msg['pattern']}",
                                 "ticker": input_msg["ticker"],
                                 "pattern": input_msg["pattern"],
                                 "pattern_num": input_msg["pattern_num"]})
        write_channel.basic_publish(exchange='uploader_exchange',
                                    routing_key=write_queue,
                                    body=output_msg)
        print(f"\nChart path for {input_msg['ticker']} sent to {write_queue}")

    read_channel.queue_declare(queue=read_queue, exclusive=False)
    read_channel.queue_bind(exchange='cb_proc_exchange', queue=read_queue, routing_key=read_queue)

    write_channel.exchange_declare(exchange='uploader_exchange', exchange_type='direct')
    write_channel.queue_declare(queue=write_queue, exclusive=False)
    write_channel.queue_bind(exchange='uploader_exchange', queue=write_queue, routing_key=write_queue)

    read_channel.basic_consume(queue=read_queue, on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for tickers. To exit press CTRL+C')

    read_channel.start_consuming()

