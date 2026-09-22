"""
Queues an async retrain via Celery and waits for it to finish. Requires a
worker already running:
celery -A src.pipelines.retrain_task worker --loglevel=info
"""

from retrain_task import retrain_model


def main():
    result = retrain_model.delay()
    print(f"Queued retrain_model task: {result.id}")
    print("Waiting for it to complete...")

    outcome = result.get(timeout=300)
    print(f"Task finished: state={result.state}, result={outcome}")


if __name__ == "__main__":
    main()
