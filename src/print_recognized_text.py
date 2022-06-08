import json

import pika

AMQP_HOSTNAME = 'localhost'


def print_recognized(ch, method, properties, body):
    data = json.loads(body)
    for text in data['texts']:
        print(text)


if __name__ == '__main__':
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=AMQP_HOSTNAME,
    ))
    channel = connection.channel()

    print('ready to print recognized texts')

    channel.basic_consume(
        queue='ocr_service_out',
        auto_ack=True,
        on_message_callback=print_recognized
    )
    channel.start_consuming()
