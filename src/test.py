import json

import pika

AMQP_HOSTNAME = 'localhost'

connection = pika.BlockingConnection(pika.ConnectionParameters(
    host=AMQP_HOSTNAME,
))
channel = connection.channel()


def recognize_text(ch, method, properties, body):
    data = json.loads(body)
    files = data['files']
    image_id = data['id']

    for file in files:
        file_info = {
            'id': image_id,
            'filepath': file['path'],
        }
        payload = json.dumps(file_info)
        channel.basic_publish(
            exchange='',
            routing_key='ocr_service_in',
            body=payload)


if __name__ == '__main__':
    data = '{"id": "d385c599-3e1d-425e-9422-f912b211f978","filepath": "../tmp/mrcnn_input/191.jpg"}'
    # data = '{"id": "d385c599-3e1d-425e-9422-f912b211f979","filepath": "../tmp/mrcnn_input/152.jpg"}'
    channel.basic_publish(
        exchange='',
        routing_key='mrcnn_service_in',
        body=data
    )

    channel.basic_consume(
        queue='mrcnn_service_out',
        auto_ack=True,
        on_message_callback=recognize_text
    )
    print('started test')
    channel.start_consuming()
