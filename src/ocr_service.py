import json

import easyocr
import pika
from pika.adapters.blocking_connection import BlockingChannel

from src.abstract_service import Service

AMQP_HOSTNAME = 'localhost'


class OcrService(Service):
    LANG_RUS = 'ru'
    LANG_EN = 'en'

    LANGUAGES = [
        LANG_RUS,
        LANG_EN,
    ]

    def __init__(self,
                 connection_channel: BlockingChannel,
                 input_queue_name: str,
                 output_queue_name: str,
                 langs: list = None,
                 ):
        langs = langs if langs is not None else self.LANGUAGES
        self.reader = easyocr.Reader(
            lang_list=langs,
            gpu=False,
        )
        print('ready to handle ocr messages')
        super().__init__(connection_channel, input_queue_name, output_queue_name)

    def execute(self, payload: bytes):
        """
        :param payload: JSON body with the following structure:
            {
              "id": "",
              "filepath": ""
            }
        :return: JSON body with the following structure:
            {
              "id": "",
              "texts": [
                "",
                ...
              ]
            }
        """
        data = json.loads(payload)
        filepath = data['filepath']
        image_id = data['id']
        results = self.reader.readtext(filepath)

        output = dict()
        output['id'] = image_id
        output['texts'] = []
        for boxes, text, accuracy in results:
            if accuracy > 0.1:
                output['texts'].append(text)
        print(f'sending: {json.dumps(output)}')
        return json.dumps(output)


if __name__ == '__main__':
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=AMQP_HOSTNAME,
    ))
    channel = connection.channel()

    service = OcrService(
        connection_channel=channel,
        input_queue_name='ocr_service_in',
        output_queue_name='ocr_service_out',
    )
