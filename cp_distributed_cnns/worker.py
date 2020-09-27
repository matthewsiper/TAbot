import os
import argparse
import json
import datetime

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pika
import boto3


buckets = ["rb-dev-ta-pattern-horrizontal2", "rb-dev-ta-pattern-wedgedown2", "rb-dev-ta-pattern-wedgeup2"]
BASE_DIR_DATA = "/Users/mattscomputer/cp_distributed_cnns/data/data{}"
BASE_DIR_MODELS = "/Users/mattscomputer/cp_distributed_cnns/models/model{}"
TRAINING_DIR = "/Users/mattscomputer/cp_distributed_cnns/data/data{}"
VALIDATION_DIR = f"/Users/mattscomputer/cp_distributed_cnns/data/data_validation"


parser = argparse.ArgumentParser(description='Worker node')
parser.add_argument('--read_queue', type=str, default='queue1')
parser.add_argument('--worker_num', type=int, default='1')

conn = boto3.client(
    's3',
    aws_access_key_id='AKIAIWPOHHSFY2F4UEOA',
    aws_secret_access_key='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'
)


def bucket_to_dir(bucket_name):
    return '_'.join(bucket_name.split('-')[2:])


def load_images(bucket_name, worker_num=1):
    for key in conn.list_objects(Bucket=bucket_name)['Contents']:
        file_name = key['Key']
        target = BASE_DIR_DATA.format(str(worker_num)) + "/{}/{}".format(bucket_to_dir(bucket_name), file_name)
        conn.download_file(bucket_name, file_name, target)

def load_image_inception(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def train_inception(worker_num):
    training_dir = TRAINING_DIR.format(str(worker_num))
    training_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='categorical',
        batch_size=10
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(300, 300),
        class_mode='categorical',
        batch_size=10
    )


    model_abs_path = BASE_DIR_MODELS.format(str(worker_num)) + f"/model{str(worker_num)}.h5"

    # Create the feature extraction model
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(300, 300, 3))

    image_model.summary()
    image_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    image_model.fit(train_generator, epochs=5, steps_per_epoch=2, validation_data=validation_generator, verbose=1,
                    validation_steps=2)

    image_model.save(model_abs_path)


def train(worker_num):
    training_dir = TRAINING_DIR.format(str(worker_num))
    training_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='categorical',
        batch_size=10
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(300, 300),
        class_mode='categorical',
        batch_size=10
    )

    model_abs_path = BASE_DIR_MODELS.format(str(worker_num)) + f"/model{str(worker_num)}.h5"


    if not os.listdir(BASE_DIR_MODELS.format(str(worker_num))):
        print(f"No model to load, creating from scratch")
        model = tf.keras.models.Sequential([
            # This is the first convolution
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
    else:
        print(f"Loading in model from previous training")
        model = load_model(model_abs_path)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit(train_generator, epochs=5, steps_per_epoch=2, validation_data=validation_generator, verbose=1,
                        validation_steps=2)

    model.save(model_abs_path)


def get_num_identifier(queue_num):
    if "1" in queue_num:
        return "1"
    elif "2" in queue_num:
        return "2"
    else:
        return "3"


if __name__ == '__main__':
    args = parser.parse_args()
    params = pika.ConnectionParameters(host='localhost',
                                       heartbeat=600,
                                       blocked_connection_timeout=300)

    connection = pika.BlockingConnection(params)
    print("connection created")

    read_channel = connection.channel()
    read_channel.exchange_declare(exchange='cnn_exchange',
                                  exchange_type='direct')
    read_queue = args.read_queue
    worker_num = args.worker_num
    read_channel.queue_declare(queue=read_queue, exclusive=False)
    read_channel.queue_bind(exchange='cnn_exchange', queue=read_queue, routing_key=read_queue)

    def callback(ch, method, properties, body):
        try:
            input_msg = json.loads(body)
            if "train" in input_msg["message"]:
                for b in buckets:
                    load_images(b, worker_num)
                # train_inception(worker_num)
                train(worker_num)

            t_diff = datetime.datetime.now() - datetime.datetime.strptime(input_msg['time'], '%Y-%m-%d %H:%M:%S.%f')
            print(f"time diff is {str(t_diff.total_seconds())}")
        except:
            msg = f"Error in worker_mq"
            raise Exception(msg)

    read_channel.basic_consume(queue=read_queue, on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')

    read_channel.start_consuming()




