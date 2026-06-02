"""Build a local knowledge-base JSONL for OKVQA evaluation.

OKVQA expects external knowledge to answer questions, and the production
pipeline retrieves evidence from a local vector store. Hitting Wikipedia
on every sample of the 5046-question validation split is too unstable
(rate limits, slow latency, intermittent network errors), so this
script pulls one short summary per OKVQA training-set ground-truth
answer and writes them as a JSONL ready to be consumed by
``rag_vqa.cli build-index``.

Usage::

    python scripts/build_okvqa_kb.py \
        --train-annotations data/okvqa/mscoco_train2014_annotations.json \
        --output data/knowledge_base/okvqa_kb.jsonl \
        --max-pages 8000

Without the ``--train-annotations`` argument the script falls back to a
small built-in seed list so it still produces a usable (but limited) KB
for smoke testing.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"
SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

SEED_TOPICS = [
    "Eiffel Tower",
    "Forbidden City",
    "Statue of Liberty",
    "Great Wall of China",
    "Pyramid of Giza",
    "Colosseum",
    "Sydney Opera House",
    "Big Ben",
    "Mount Fuji",
    "Pizza",
    "Sushi",
    "Hamburger",
    "Apple",
    "Banana",
    "Bicycle",
    "Microwave oven",
    "Stop sign",
    "Traffic light",
    "Fire extinguisher",
    "Umbrella",
    "iPhone",
    "MacBook",
    "Tesla Model 3",
    "Giant panda",
    "Lion",
    "Penguin",
    "Owl",
    "Dog",
    "Cat",
    "Horse",
    "Cow",
    "Elephant",
    "Bear",
    "Giraffe",
    "Surfboard",
    "Skateboard",
    "Snowboard",
    "Tennis racket",
    "Baseball bat",
    "Frisbee",
    "Pizza oven",
    "Refrigerator",
    "Toaster",
    "Hair drier",
    "Cell phone",
    "Laptop",
    "Keyboard",
    "Computer mouse",
    "Television",
    "Remote control",
    "Bed",
    "Couch",
    "Dining table",
    "Toilet",
    "Sink",
    "Vase",
    "Wine glass",
    "Cup",
    "Fork",
    "Knife",
    "Spoon",
    "Bowl",
    "Bottle",
    "Carrot",
    "Broccoli",
    "Donut",
    "Cake",
    "Sandwich",
    "Hot dog",
    "Orange",
    "Bus",
    "Car",
    "Truck",
    "Train",
    "Boat",
    "Airplane",
    "Motorcycle",
    "Hot air balloon",
    "Aircraft carrier",
    "Helicopter",
    "Bench",
    "Backpack",
    "Handbag",
    "Suitcase",
    "Tie",
    "Book",
    "Clock",
    "Chair",
    "Potted plant",
    "Rose",
    "Bonsai",
]


def collect_topics(args: argparse.Namespace) -> list[str]:
    topics: list[str] = []
    seen: set[str] = set()

    def push(name: str) -> None:
        cleaned = name.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        topics.append(cleaned)

    for seed in SEED_TOPICS:
        push(seed)

    if args.train_annotations:
        with Path(args.train_annotations).open("r", encoding="utf-8") as f:
            blob = json.load(f)
        counter: Counter[str] = Counter()
        for ann in blob.get("annotations", []):
            for a in ann.get("answers", []):
                ans = (a.get("answer") or "").strip()
                if not ans or len(ans) <= 1:
                    continue
                counter[ans.lower()] += 1
        for word, _ in counter.most_common(args.max_pages):
            push(word)
    return topics[: args.max_pages]


def fetch_summary(session: requests.Session, title: str, timeout: int) -> dict | None:
    try:
        url = SUMMARY_API.format(title=title.replace(" ", "_"))
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("type") == "disambiguation":
            return None
        extract = (data.get("extract") or "").strip()
        if not extract:
            return None
        return {"title": data.get("title") or title, "extract": extract}
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--train-annotations",
        default=None,
        help="Path to OKVQA train annotations JSON. When given, the script harvests "
        "high frequency answer strings to grow the topic list.",
    )
    parser.add_argument("--output", default="data/knowledge_base/okvqa_kb.jsonl")
    parser.add_argument("--max-pages", type=int, default=200, help="Cap on Wikipedia summaries to fetch")
    parser.add_argument("--timeout", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0, help="Sleep between Wikipedia calls")
    args = parser.parse_args()

    topics = collect_topics(args)
    print(f"Collected {len(topics)} candidate topics")

    session = requests.Session()
    session.headers.update({"User-Agent": "RAG-VQA/0.1 (educational; OKVQA KB build)"})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, topic in enumerate(topics, start=1):
            summary = fetch_summary(session, topic, timeout=args.timeout)
            if not summary:
                continue
            doc = {
                "id": f"wiki_{idx:05d}",
                "title": summary["title"],
                "text": summary["extract"],
                "source": f"https://en.wikipedia.org/wiki/{summary['title'].replace(' ', '_')}",
                "type": "text",
                "image_path": None,
                "tags": [summary["title"].lower()],
                "metadata": {"language": "en", "topic": topic},
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
            if idx % 100 == 0:
                print(f"  fetched {idx}/{len(topics)}, saved {written}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    print(f"Wrote {written} documents to {out_path}")


if __name__ == "__main__":
    main()
