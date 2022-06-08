import json

from PIL import Image
import numpy as np
import pika
import pytesseract
from pika.adapters.blocking_connection import BlockingChannel
from skimage import filters, io, util

from src.abstract_service import Service

AMQP_HOSTNAME = 'localhost'


class OcrService(Service):
    LANG_RUS = 'rus'
    LANG_EN = 'eng'

    LANGUAGES = [
        LANG_RUS,
        LANG_EN,
    ]

    def __init__(self,
                 connection_channel: BlockingChannel,
                 input_queue_name: str,
                 output_queue_name: str,
                 lang: str = 'rus',
                 ):
        assert lang in self.LANGUAGES
        self.lang = lang
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

        image = io.imread(filepath)
        thresh = filters.threshold_mean(image)
        threshold_image = image > thresh
        im = Image.fromarray((threshold_image * 255).astype(np.uint8))
        text = pytesseract.image_to_string(im, lang=self.lang)

        output = dict()
        output['id'] = image_id
        output['texts'] = text.split('\n')
        print(f'sending: {json.dumps(output)}')
        return json.dumps(output)


if __name__ == '__main__':
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=AMQP_HOSTNAME,
    ))
    channel = connection.channel()

    print('ready to handle ocr messages')

    service = OcrService(
        connection_channel=channel,
        input_queue_name='ocr_service_in',
        output_queue_name='ocr_service_out',
        lang=OcrService.LANG_RUS,
    )
