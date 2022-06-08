import json
import os
import pathlib

import pika
from skimage import io
from mrcnn.model import MaskRCNN
from pika.adapters.blocking_connection import BlockingChannel

from src.abstract_service import Service
from src.cnn import InferenceConfig, OfficeObjectDataset

AMQP_HOSTNAME = 'localhost'
MODEL_NAME = 'offigator-20_epochs.h5'
TMP_FILES_DIRECTORY = '../tmp'


class MrcnnService(Service):

    def __init__(self,
                 connection_channel: BlockingChannel,
                 input_queue_name: str,
                 output_queue_name: str,
                 mrcnn_model: MaskRCNN,
                 output_directory: str,
                 ):
        self.mrcnn_model = mrcnn_model
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
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
              "files": [
                {
                  "path": "",
                  "class": ""
                }
              ]
            }
        """
        data = json.loads(payload)
        filepath = data['filepath']
        image_id = data['id']

        image = io.imread(filepath)
        results = self.mrcnn_model.detect([image], verbose=1)
        r = results[0]
        output = dict()
        output['id'] = image_id
        output['files'] = []
        if not os.path.exists(f'{self.output_directory}/{image_id}'):
            os.makedirs(f'{self.output_directory}/{image_id}')
        for i, box in enumerate(r['rois']):
            x1, y1, x2, y2 = box
            cropped_image = image[x1:x2, y1:y2]
            path = f'{self.output_directory}/{image_id}/{i}.jpg'
            io.imsave(path, cropped_image)
            output['files'].append({'path': path, 'class': OfficeObjectDataset.OBJ_TYPES[r['class_ids'][i] - 1]})
        print(f'sending: {json.dumps(output)}')
        return json.dumps(output)


if __name__ == '__main__':
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=AMQP_HOSTNAME,
    ))
    channel = connection.channel()

    current_dir = pathlib.Path.cwd()
    inference_cfg = InferenceConfig()
    inference_model = MaskRCNN(mode='inference', model_dir=f'{current_dir}/../models', config=inference_cfg)
    last_weights = f'{current_dir}/../models/{MODEL_NAME}'
    inference_model.load_weights(last_weights, by_name=True)

    print('ready to handle mcrnn messages')

    service = MrcnnService(
        connection_channel=channel,
        input_queue_name='mrcnn_service_in',
        output_queue_name='mrcnn_service_out',
        mrcnn_model=inference_model,
        output_directory=f'{TMP_FILES_DIRECTORY}/mrcnn_output',
    )
