import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
channel = connection.channel()

channel.queue_declare(queue="hello_queue")# we created a queue here declaring an existing queue doesn't create a duplicate. It essentially ensures that the queue exists.

channel.basic_publish(exchange="", routing_key="hello_queue", body="Hello, world!")
print("Sent 'Hello, world!'")

connection.close()
