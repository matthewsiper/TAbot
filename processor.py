import os
from chart_builder_worker import ChartBuilderWorker
import pika
from constants import FINVIZ_URLS, CBUILDER_WORKER_QUEUES
import time
import json
import datetime as dt
from helpers import get_pattern_from_url

params = pika.ConnectionParameters(host='localhost',
                                   heartbeat=600,
                                   blocked_connection_timeout=300)

connection = pika.BlockingConnection(params)
channel = connection.channel()
channel.exchange_declare(exchange='cb_proc_exchange',
                         exchange_type='direct')

for queue in CBUILDER_WORKER_QUEUES:
    channel.queue_declare(queue=queue, exclusive=False)
    channel.queue_bind(
        exchange='cb_proc_exchange', queue=queue, routing_key=queue)


def build_message(pattern, ticker):
    return json.dumps({"pattern": pattern, "ticker": ticker})

cb = ChartBuilderWorker()

if __name__ == '__main__':
    for queue, pattern_tickers_dict_list in FINVIZ_URLS.items():
        file_num = 1
        for patter_tickers_dict in pattern_tickers_dict_list:
            for pattern, tickers_list in patter_tickers_dict.items():
                for ticker in tickers_list:
                    try:
                        channel.basic_publish(exchange='cb_proc_exchange',
                                              routing_key=queue,
                                              body=json.dumps({"ticker": ticker,
                                                               "pattern": pattern,
                                                               "pattern_num": file_num}))
                        print(f"running worker for {ticker} and pattern {pattern}")
                        file_num += 1
                    except:
                        raise Exception("error")

