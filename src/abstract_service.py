import abc

from pika.adapters.blocking_connection import BlockingChannel


class Service(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 connection_channel: BlockingChannel,
                 input_queue_name: str,
                 output_queue_name: str,
                 ):
        self.input_queue_name = input_queue_name
        self.output_queue_name = output_queue_name
        self.channel = connection_channel

        self.channel.queue_declare(queue=self.input_queue_name)
        self.channel.queue_declare(queue=self.output_queue_name)
        self.channel.basic_consume(
            queue=self.input_queue_name,
            auto_ack=True,
            on_message_callback=self.consume
        )
        self.channel.start_consuming()

    def consume(self, ch, method, properties, body):
        payload = self.execute(
            payload=body
        )
        self.channel.basic_publish(
            exchange='',
            routing_key=self.output_queue_name,
            body=payload)

    @abc.abstractmethod
    def execute(self, payload: bytes):
        """
        Override for specific service
        :param payload:
        :return bytes payload:
        """
        pass
