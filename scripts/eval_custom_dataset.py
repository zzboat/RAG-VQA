from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_vqa.config import Settings
from rag_vqa.eval.okvqa_eval import extract_short_answer, normalize_answer
from rag_vqa.pipeline import RAGVQAPipeline
from rag_vqa.retriever import KnowledgeBase


def load_custom_dataset(dataset_path: str | Path, images_base: str | Path) -> list[dict]:
    dataset_path = Path(dataset_path)
    images_base = Path(images_base)
    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for item in raw:
        image_path = images_base / item["image"]
        if not image_path.exists():
            print(f"[warn] image not found: {image_path}, skipping")
            continue
        samples.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "image_path": str(image_path),
            }
        )
    return samples


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def is_visual_only(question: str) -> bool:
    visual_keywords = [
        "what color", "how many", "what fruit", "what food",
        "what animal", "what drink", "what vehicle", "what sport",
        "what is the girl", "holding", "on the bed", "on the plate",
        "in the cup", "traffic light", "cat in the image",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in visual_keywords)


def main():
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/sunye_2/zhangchuhan/dataset/rag-vqa/rag_vqa_question_answer_image.json"
    )
    images_base = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "/home/sunye_2/zhangchuhan/dataset/rag-vqa"
    )
    kb_path = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "data/knowledge_base/custom_kb.jsonl"
    index_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJECT_ROOT / "outputs/custom_index"
    output_path = Path(sys.argv[5]) if len(sys.argv) > 5 else PROJECT_ROOT / "outputs/custom_eval.json"
    enable_web = "--web" in sys.argv

    samples = load_custom_dataset(dataset_path, images_base)
    print(f"Loaded {len(samples)} samples")

    settings = Settings(top_k=5, debug=False)
    kb = KnowledgeBase.from_jsonl(str(kb_path), settings=settings)
    kb.save(str(index_dir))
    print(f"Built index with {len(kb.docs)} documents")

    pipeline = RAGVQAPipeline(kb=kb, settings=settings, enable_web=enable_web)

    results = {"visual_only": [], "knowledge": [], "all": []}
    preds = []
    total_baseline = 0.0
    total_rag = 0.0

    num_visual = 0
    num_knowledge = 0
    start = time.time()

    for idx, sample in enumerate(samples, start=1):
        try:
            answer_obj = pipeline.ask(sample["image_path"], sample["question"], top_k=5)

            baseline_pred = normalize_answer(answer_obj.visual_answer or "")
            baseline_score = exact_match(baseline_pred, sample["answer"])

            rag_pred_short = extract_short_answer(answer_obj.answer)
            rag_score = exact_match(rag_pred_short, sample["answer"])

            category = "visual_only" if is_visual_only(sample["question"]) else "knowledge"
            if category == "visual_only":
                num_visual += 1
            else:
                num_knowledge += 1

            total_baseline += baseline_score
            total_rag += rag_score

            results[category].append(
                {
                    "baseline_score": baseline_score,
                    "rag_score": rag_score,
                }
            )
            results["all"].append(
                {
                    "baseline_score": baseline_score,
                    "rag_score": rag_score,
                }
            )

            preds.append(
                {
                    "idx": idx,
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "baseline_answer": answer_obj.visual_answer or "",
                    "rag_answer": answer_obj.answer,
                    "rag_short_answer": rag_pred_short,
                    "baseline_score": baseline_score,
                    "rag_score": rag_score,
                    "category": category,
                }
            )

        except Exception as exc:
            preds.append(
                {
                    "idx": idx,
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "error": repr(exc),
                    "category": is_visual_only(sample["question"]) and "visual_only" or "knowledge",
                    "baseline_score": 0.0,
                    "rag_score": 0.0,
                }
            )
            print(f"[error] sample {idx}: {exc}")

        if idx % 5 == 0:
            n = idx
            print(
                f"[progress] {idx}/{len(samples)} "
                f"baseline_acc={total_baseline / n * 100:.1f}% "
                f"rag_acc={total_rag / n * 100:.1f}%",
                flush=True,
            )

    elapsed = time.time() - start
    n = len(samples)

    def acc(score_sum, count):
        return score_sum / count if count else 0.0

    summary = {
        "num_samples": n,
        "elapsed_seconds": elapsed,
        "baseline": {
            "overall_accuracy": acc(total_baseline, n),
            "visual_only_accuracy": acc(
                sum(r["baseline_score"] for r in results["visual_only"]), num_visual
            ),
            "knowledge_accuracy": acc(
                sum(r["baseline_score"] for r in results["knowledge"]), num_knowledge
            ),
        },
        "rag": {
            "overall_accuracy": acc(total_rag, n),
            "visual_only_accuracy": acc(
                sum(r["rag_score"] for r in results["visual_only"]), num_visual
            ),
            "knowledge_accuracy": acc(
                sum(r["rag_score"] for r in results["knowledge"]), num_knowledge
            ),
        },
        "improvement": {
            "overall_pts": acc(total_rag, n) - acc(total_baseline, n),
            "knowledge_pts": acc(
                sum(r["rag_score"] for r in results["knowledge"]), num_knowledge
            )
            - acc(sum(r["baseline_score"] for r in results["knowledge"]), num_knowledge),
        },
        "predictions": preds,
    }

    print("\n" + "=" * 60)
    print("RAG-VQA Custom Dataset Evaluation")
    print("=" * 60)
    print(f"  Samples: {n}  (visual-only: {num_visual}, knowledge: {num_knowledge})")
    print(f"  Time:    {elapsed:.1f}s")
    print()
    print("  Overall Accuracy:")
    print(f"    Baseline (BLIP-VQA):  {summary['baseline']['overall_accuracy'] * 100:.1f}%")
    print(f"    RAG-VQA:              {summary['rag']['overall_accuracy'] * 100:.1f}%")
    print(f"    Improvement:          +{summary['improvement']['overall_pts'] * 100:.1f} pts")
    print()
    print("  Visual-Only Questions:")
    print(f"    Baseline (BLIP-VQA):  {summary['baseline']['visual_only_accuracy'] * 100:.1f}%")
    print(f"    RAG-VQA:              {summary['rag']['visual_only_accuracy'] * 100:.1f}%")
    print()
    print("  Knowledge Questions:")
    print(f"    Baseline (BLIP-VQA):  {summary['baseline']['knowledge_accuracy'] * 100:.1f}%")
    print(f"    RAG-VQA:              {summary['rag']['knowledge_accuracy'] * 100:.1f}%")
    print(f"    Improvement:          +{summary['improvement']['knowledge_pts'] * 100:.1f} pts")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
