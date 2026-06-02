"""End-to-end OKVQA evaluation for the RAG-VQA pipeline.

This module loads OKVQA v1.1 questions / annotations and runs the
``RAGVQAPipeline`` on each sample, then scores the predicted free-form
answer using the standard VQA soft-accuracy metric:

    acc(p) = mean over the 10 annotators of min(#match_in_other_9 / 3, 1)

Only a small subset of the official ``vqaEval.py`` post-processing is
needed (lower-casing, contraction expansion, article / punctuation
removal, and number-word normalisation) because OKVQA reuses the
VQAv2 evaluation toolkit.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..config import Settings
from ..pipeline import RAGVQAPipeline
from ..retriever import KnowledgeBase


# --- Answer normalisation helpers (subset of official vqaEval.py) -----------

_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "id": "I'd",
    "im": "I'm",
    "ive": "I've",
    "isnt": "isn't",
    "its": "it's",
    "lets": "let's",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "whos": "who's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ARTICLES = {"a", "an", "the"}

_PUNCT = ";/[]\"{}()=+\\_-><@`,?!"
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")


def _process_punctuation(text: str) -> str:
    out = text
    for ch in _PUNCT:
        if ch in out or "'" in out or "\"" in out:
            out = out.replace(ch, "")
    out = _PERIOD_STRIP.sub("", out, re.UNICODE)
    return out


def _process_digit_article(text: str) -> str:
    out_tokens: list[str] = []
    for token in text.lower().split():
        token = _NUMBER_WORDS.get(token, token)
        if token in _ARTICLES:
            continue
        out_tokens.append(_CONTRACTIONS.get(token, token))
    return " ".join(out_tokens)


def normalize_answer(answer: str) -> str:
    """Replicate the OKVQA / VQA reference answer cleanup."""

    text = answer.replace("\n", " ").replace("\t", " ").strip()
    text = _COMMA_STRIP.sub(r"\1\3", text)
    text = _process_punctuation(text)
    text = _process_digit_article(text)
    return text


def soft_accuracy(prediction: str, ground_truth_answers: list[str]) -> float:
    """Standard VQA soft-accuracy with 10 leave-one-out subsets."""

    if not ground_truth_answers:
        return 0.0
    pred = normalize_answer(prediction)
    gts = [normalize_answer(a) for a in ground_truth_answers]
    accs: list[float] = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1 :]
        matches = sum(1 for o in others if o == pred)
        accs.append(min(matches / 3.0, 1.0))
    return sum(accs) / len(accs)


# --- OKVQA loading ---------------------------------------------------------


@dataclass
class OKVQASample:
    question_id: int
    image_id: int
    question: str
    answers: list[str]
    image_path: str
    question_type: str = ""
    answer_type: str = ""


def load_okvqa(
    questions_path: str | Path,
    annotations_path: str | Path,
    images_dir: str | Path,
    image_prefix: str = "COCO_val2014_",
    image_suffix: str = ".jpg",
) -> list[OKVQASample]:
    questions_path = Path(questions_path)
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)

    with questions_path.open("r", encoding="utf-8") as f:
        questions_blob = json.load(f)
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations_blob = json.load(f)

    annotations_by_qid = {ann["question_id"]: ann for ann in annotations_blob["annotations"]}

    samples: list[OKVQASample] = []
    for q in questions_blob["questions"]:
        ann = annotations_by_qid.get(q["question_id"])
        if ann is None:
            continue
        image_id = q["image_id"]
        image_name = f"{image_prefix}{image_id:012d}{image_suffix}"
        samples.append(
            OKVQASample(
                question_id=q["question_id"],
                image_id=image_id,
                question=q["question"],
                answers=[a["answer"] for a in ann.get("answers", [])],
                image_path=str(images_dir / image_name),
                question_type=ann.get("question_type", ""),
                answer_type=ann.get("answer_type", ""),
            )
        )
    return samples


# --- Evaluation core -------------------------------------------------------


@dataclass
class EvaluationResult:
    overall_accuracy: float
    num_samples: int
    by_question_type: dict[str, dict[str, float]] = field(default_factory=dict)
    by_answer_type: dict[str, dict[str, float]] = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def extract_short_answer(answer: str) -> str:
    """Pull out a short answer phrase from the pipeline's final string.

    The generator is asked to produce 2-3 sentences, but VQA scoring
    expects a short phrase. We take the first sentence, strip parenthesised
    citation markers like ``[1]`` or trailing punctuation, and cap the
    result at five tokens which is what OKVQA answers usually are.
    """

    text = answer.strip()
    if not text:
        return ""
    text = re.split(r"[.?!\n。！？]", text, maxsplit=1)[0]
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" :;-—,.")
    tokens = text.split()
    if len(tokens) > 5:
        tokens = tokens[:5]
    return " ".join(tokens).lower()


def _accumulate(stats: dict[str, dict[str, float]], key: str, score: float) -> None:
    bucket = stats.setdefault(key, {"sum": 0.0, "count": 0.0})
    bucket["sum"] += score
    bucket["count"] += 1.0


def _finalize(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, bucket in stats.items():
        count = bucket["count"]
        out[key] = {
            "accuracy": (bucket["sum"] / count) if count else 0.0,
            "num_samples": count,
        }
    return out


def evaluate(
    samples: Iterable[OKVQASample],
    pipeline: RAGVQAPipeline,
    *,
    top_k: int = 5,
    limit: int | None = None,
    progress_every: int = 25,
) -> EvaluationResult:
    samples = list(samples)
    if limit is not None:
        samples = samples[:limit]

    by_q_type: dict[str, dict[str, float]] = {}
    by_a_type: dict[str, dict[str, float]] = {}
    predictions: list[dict] = []
    total_score = 0.0
    start = time.time()
    for idx, sample in enumerate(samples, start=1):
        try:
            answer_obj = pipeline.ask(sample.image_path, sample.question, top_k=top_k)
            short_answer = extract_short_answer(answer_obj.answer)
        except Exception as exc:
            short_answer = ""
            answer_obj = None
            err_repr = repr(exc)
            predictions.append(
                {
                    "question_id": sample.question_id,
                    "image_id": sample.image_id,
                    "question": sample.question,
                    "raw_answer": "",
                    "predicted": "",
                    "score": 0.0,
                    "answers": sample.answers,
                    "error": err_repr,
                }
            )
            _accumulate(by_q_type, sample.question_type or "unknown", 0.0)
            _accumulate(by_a_type, sample.answer_type or "unknown", 0.0)
            continue

        score = soft_accuracy(short_answer, sample.answers)
        total_score += score
        _accumulate(by_q_type, sample.question_type or "unknown", score)
        _accumulate(by_a_type, sample.answer_type or "unknown", score)
        predictions.append(
            {
                "question_id": sample.question_id,
                "image_id": sample.image_id,
                "question": sample.question,
                "raw_answer": answer_obj.answer if answer_obj else "",
                "predicted": short_answer,
                "score": score,
                "answers": sample.answers,
            }
        )

        if progress_every and idx % progress_every == 0:
            running = total_score / idx
            print(f"[okvqa-eval] {idx}/{len(samples)} running_acc={running * 100:.2f}", flush=True)

    elapsed = time.time() - start
    n = len(samples) or 1
    return EvaluationResult(
        overall_accuracy=total_score / n,
        num_samples=len(samples),
        by_question_type=_finalize(by_q_type),
        by_answer_type=_finalize(by_a_type),
        predictions=predictions,
        elapsed_seconds=elapsed,
    )


def run(
    questions_path: str | Path,
    annotations_path: str | Path,
    images_dir: str | Path,
    kb: KnowledgeBase,
    output_path: str | Path | None = None,
    *,
    settings: Settings | None = None,
    enable_web: bool = False,
    top_k: int = 5,
    limit: int | None = None,
) -> EvaluationResult:
    samples = load_okvqa(questions_path, annotations_path, images_dir)
    pipeline = RAGVQAPipeline(kb=kb, settings=settings, enable_web=enable_web)
    result = evaluate(samples, pipeline, top_k=top_k, limit=limit)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "overall_accuracy": result.overall_accuracy,
                    "overall_accuracy_pct": result.overall_accuracy * 100,
                    "num_samples": result.num_samples,
                    "by_question_type": result.by_question_type,
                    "by_answer_type": result.by_answer_type,
                    "elapsed_seconds": result.elapsed_seconds,
                    "predictions": result.predictions,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    return result
