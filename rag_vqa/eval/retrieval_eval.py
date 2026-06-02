"""Retrieval coverage evaluation: question-only baseline vs joint image-text query.

For each (question, optional image, gold_doc_ids) sample, we run the local
``KnowledgeBase`` retriever twice:

* baseline -- only the raw question is used as the text query.
* full     -- BLIP caption + question-aware keywords are joined into the
              text query, exactly the same way ``QueryGenerator`` does in
              the production pipeline.

We then compute Recall@k and Hit@k over k in {1, 3, 5, 10} and report the
absolute and relative improvement, so a number such as
``Recall@5: 0.42 -> 0.61`` can be cited on the resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..config import Settings
from ..query import QueryGenerator
from ..retriever import KnowledgeBase
from ..schemas import QueryBundle


DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class RetrievalSample:
    qid: str
    question: str
    gold_doc_ids: list[str]
    image: str | None = None
    synthetic_caption: str | None = None


def load_eval_set(path: str | Path) -> list[RetrievalSample]:
    samples: list[RetrievalSample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            samples.append(
                RetrievalSample(
                    qid=str(item["qid"]),
                    question=str(item["question"]),
                    gold_doc_ids=list(item["gold_doc_ids"]),
                    image=item.get("image"),
                    synthetic_caption=item.get("synthetic_caption"),
                )
            )
    return samples


def baseline_query(sample: RetrievalSample) -> QueryBundle:
    return QueryBundle(
        question=sample.question,
        visual_caption="",
        text_query=sample.question,
        keywords=[],
    )


def full_query(
    sample: RetrievalSample,
    caption_fn: Callable[[str], str] | None = None,
    query_generator: QueryGenerator | None = None,
) -> QueryBundle:
    """Build the joint image-text query used in production.

    ``caption_fn`` is invoked when ``sample.image`` is set and no
    ``synthetic_caption`` is provided, so unit tests can inject a fake
    captioner without spinning up BLIP.
    """

    query_generator = query_generator or QueryGenerator()
    if sample.synthetic_caption:
        caption = sample.synthetic_caption
    elif sample.image and caption_fn is not None:
        caption = caption_fn(sample.image) or ""
    else:
        caption = ""
    return query_generator.generate(sample.question, caption)


def _retrieve_doc_ids(
    kb: KnowledgeBase,
    query: QueryBundle,
    image_path: str | Path,
    top_k: int,
) -> list[str]:
    evidences = kb.retrieve(query, image_path, top_k=top_k)
    return [ev.id for ev in evidences]


def _hit_recall(predicted_ids: list[str], gold_ids: list[str], k: int) -> tuple[float, float]:
    top = predicted_ids[:k]
    gold_set = set(gold_ids)
    hit = 1.0 if any(pid in gold_set for pid in top) else 0.0
    if gold_set:
        recall = sum(1 for pid in top if pid in gold_set) / len(gold_set)
    else:
        recall = 0.0
    return hit, recall


def evaluate(
    samples: Iterable[RetrievalSample],
    kb: KnowledgeBase,
    *,
    ks: Iterable[int] = DEFAULT_KS,
    caption_fn: Callable[[str], str] | None = None,
    placeholder_image: str | Path | None = None,
) -> dict:
    """Run baseline and full retrieval, returning aggregated metrics.

    ``placeholder_image`` is passed to ``KnowledgeBase.retrieve``; using a
    non-existent path forces the retriever to return zero image-scores so
    the ablation focuses purely on the text query construction.
    """

    samples = list(samples)
    ks = list(ks)
    max_k = max(ks)

    placeholder = Path(placeholder_image or "__rag_vqa_no_image__.png")

    per_sample: list[dict] = []
    sums = {
        mode: {f"hit@{k}": 0.0 for k in ks} | {f"recall@{k}": 0.0 for k in ks}
        for mode in ("baseline", "full")
    }

    qg = QueryGenerator()
    for sample in samples:
        baseline_q = baseline_query(sample)
        full_q = full_query(sample, caption_fn=caption_fn, query_generator=qg)

        baseline_ids = _retrieve_doc_ids(kb, baseline_q, placeholder, top_k=max_k)
        full_ids = _retrieve_doc_ids(kb, full_q, placeholder, top_k=max_k)

        sample_metrics: dict = {
            "qid": sample.qid,
            "question": sample.question,
            "gold_doc_ids": sample.gold_doc_ids,
            "baseline_top": baseline_ids,
            "full_top": full_ids,
            "baseline_text_query": baseline_q.text_query,
            "full_text_query": full_q.text_query,
        }
        for k in ks:
            for mode, ids in (("baseline", baseline_ids), ("full", full_ids)):
                hit, recall = _hit_recall(ids, sample.gold_doc_ids, k)
                sample_metrics[f"{mode}_hit@{k}"] = hit
                sample_metrics[f"{mode}_recall@{k}"] = recall
                sums[mode][f"hit@{k}"] += hit
                sums[mode][f"recall@{k}"] += recall
        per_sample.append(sample_metrics)

    n = max(len(samples), 1)
    aggregated = {mode: {key: value / n for key, value in metrics.items()} for mode, metrics in sums.items()}

    deltas: dict[str, dict[str, float]] = {}
    for k in ks:
        for prefix in ("hit", "recall"):
            key = f"{prefix}@{k}"
            base = aggregated["baseline"][key]
            full = aggregated["full"][key]
            deltas.setdefault(key, {})
            deltas[key]["baseline"] = base
            deltas[key]["full"] = full
            deltas[key]["delta_pts"] = full - base
            deltas[key]["relative_improvement"] = (full - base) / base if base > 0 else None

    return {
        "num_samples": len(samples),
        "ks": ks,
        "aggregated": aggregated,
        "deltas": deltas,
        "per_sample": per_sample,
    }


def run(
    eval_path: str | Path,
    kb: KnowledgeBase,
    output_path: str | Path | None = None,
    *,
    caption_fn: Callable[[str], str] | None = None,
) -> dict:
    samples = load_eval_set(eval_path)
    result = evaluate(samples, kb, caption_fn=caption_fn)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def make_eval_kb(kb_path: str | Path, *, settings: Settings | None = None) -> KnowledgeBase:
    """Helper that loads a KB with retrieval-eval friendly settings.

    We zero out the image weight (the eval focuses on text query
    construction) and disable the min-score filter so the full ranked
    list is available for Recall@10.
    """

    base = settings or Settings()
    eval_settings = Settings(
        text_embedding_model=base.text_embedding_model,
        image_embedding_model=base.image_embedding_model,
        caption_model=base.caption_model,
        vqa_model=base.vqa_model,
        generator_model=base.generator_model,
        device=base.device,
        vision_local_files_only=base.vision_local_files_only,
        top_k=10,
        text_weight=1.0,
        image_weight=0.0,
        min_evidence_score=-1.0,
        web_timeout=base.web_timeout,
        web_use_env_proxy=base.web_use_env_proxy,
        enable_generator=False,
        enable_blip_vqa=False,
        debug=base.debug,
    )
    return KnowledgeBase.from_jsonl(kb_path, settings=eval_settings)


def format_summary(result: dict) -> str:
    """Render a short human readable summary of the metric table."""

    lines = [f"# samples: {result['num_samples']}", ""]
    header = f"{'metric':<12}{'baseline':>12}{'full':>12}{'delta':>12}{'rel':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, payload in result["deltas"].items():
        rel = payload["relative_improvement"]
        rel_str = "n/a" if rel is None else f"{rel * 100:+.1f}%"
        lines.append(
            f"{key:<12}{payload['baseline']:>12.4f}{payload['full']:>12.4f}"
            f"{payload['delta_pts']:>+12.4f}{rel_str:>12}"
        )
    return "\n".join(lines)
