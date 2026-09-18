"""
Reads the synthetic event CSVs from data/synthetic/ and publishes each row
as a JSON message to RabbitMQ, one durable queue per event type.
"""

import csv
import json

import pika

QUEUES = ["login_events", "support_tickets", "feature_usage_logs"]
CSV_DIR = "data/synthetic"


def coerce(value):
    """Turn a raw CSV string into bool/int/float/None where possible, else leave as str."""
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def publish_csv(channel, queue_name):
    csv_path = f"{CSV_DIR}/{queue_name}.csv"
    count = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            message = {key: coerce(value) for key, value in row.items()}
            channel.basic_publish(
                exchange="",
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2),  # persistent
            )
            count += 1

    print(f"Published {count} messages to '{queue_name}' from {csv_path}")


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()

    for queue_name in QUEUES:
        channel.queue_declare(queue=queue_name, durable=True)

    for queue_name in QUEUES:
        publish_csv(channel, queue_name)

    connection.close()


if __name__ == "__main__":
    main()
