"""
Prerequisite: the FastAPI serving app must already be running:
uvicorn src.serving.api:app --reload --port 8000
(predict_churn_tool calls http://localhost:8000/predict over HTTP)
"""

import sys

from agent import agent

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252,
# which can't print characters some LLM output contains (e.g. non-breaking hyphens)

QUESTION = "What's the churn risk for customer 7590-VHVEG?"


def main():
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})

    print(f"Question: {QUESTION}\n")
    print("=== Full message history (tool calls + args included) ===\n")
    for msg in result["messages"]:
        msg.pretty_print()
        print()


if __name__ == "__main__":
    main()
