from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .debug import debug_dump
from .eval import okvqa_eval, retrieval_eval
from .pipeline import RAGVQAPipeline
from .retriever import KnowledgeBase


def build_index(args: argparse.Namespace) -> None:
    settings = Settings(debug=args.debug)
    debug_dump(settings, "cli.build_index.args", vars(args))
    kb = KnowledgeBase.from_jsonl(args.kb, settings=settings)
    kb.save(args.index_dir)
    print(f"Built index with {len(kb.docs)} documents: {args.index_dir}")


def ask(args: argparse.Namespace) -> None:
    settings = Settings(top_k=args.top_k, debug=args.debug)
    debug_dump(settings, "cli.ask.args", vars(args))
    index_dir = Path(args.index_dir)
    if index_dir.exists() and (index_dir / "documents.json").exists():
        kb = KnowledgeBase.load(index_dir, settings=settings)
        debug_dump(
            settings,
            "index.load",
            {
                "index_dir": str(index_dir),
                "doc_count": len(kb.docs),
                "text_vector_shape": kb.text_vectors.shape,
                "image_vector_shape": kb.image_vectors.shape,
            },
        )
    else:
        kb = KnowledgeBase.from_jsonl(args.kb, settings=settings)
        kb.save(index_dir)

    pipeline = RAGVQAPipeline(kb=kb, settings=settings, enable_web=args.web)
    result = pipeline.ask(args.image, args.question, top_k=args.top_k)
    payload = asdict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def eval_retrieval(args: argparse.Namespace) -> None:
    settings = Settings(
        text_weight=1.0,
        image_weight=0.0,
        min_evidence_score=-1.0,
        enable_generator=False,
        enable_blip_vqa=False,
        debug=args.debug,
    )
    debug_dump(settings, "cli.eval_retrieval.args", vars(args))

    index_dir = Path(args.index_dir) if args.index_dir else None
    if index_dir and (index_dir / "documents.json").exists():
        kb = KnowledgeBase.load(index_dir, settings=settings)
    else:
        kb = KnowledgeBase.from_jsonl(args.kb, settings=settings)

    caption_fn = None
    if args.use_blip_caption:
        from .vision import ImageDescriber

        describer = ImageDescriber(settings.caption_model, settings=settings)
        caption_fn = describer.describe

    result = retrieval_eval.run(args.eval, kb, output_path=args.output, caption_fn=caption_fn)
    print(retrieval_eval.format_summary(result))
    if args.output:
        print(f"\nWrote per-sample report to {args.output}")


def eval_okvqa(args: argparse.Namespace) -> None:
    settings = Settings(
        top_k=args.top_k,
        debug=args.debug,
        cache_caption=args.cache_caption,
        caption_cache_path=args.caption_cache,
    )
    debug_dump(settings, "cli.eval_okvqa.args", vars(args))

    index_dir = Path(args.index_dir)
    if (index_dir / "documents.json").exists():
        kb = KnowledgeBase.load(index_dir, settings=settings)
    else:
        kb = KnowledgeBase.from_jsonl(args.kb, settings=settings)

    result = okvqa_eval.run(
        args.questions,
        args.annotations,
        args.images_dir,
        kb,
        output_path=args.output,
        settings=settings,
        enable_web=args.web,
        top_k=args.top_k,
        limit=args.limit,
    )
    print(
        f"OKVQA accuracy: {result.overall_accuracy * 100:.2f}  "
        f"(N={result.num_samples}, elapsed={result.elapsed_seconds:.1f}s)"
    )
    if args.output:
        print(f"Wrote detailed report to {args.output}")


def serve(args: argparse.Namespace) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise SystemExit("Please install gradio first: pip install gradio") from exc

    settings = Settings(top_k=args.top_k, debug=args.debug)
    debug_dump(settings, "cli.serve.args", vars(args))
    index_dir = Path(args.index_dir)
    kb = KnowledgeBase.load(index_dir, settings=settings) if (index_dir / "documents.json").exists() else KnowledgeBase.from_jsonl(args.kb, settings)
    pipeline = RAGVQAPipeline(kb=kb, settings=settings, enable_web=args.web)

    def infer(image, question):
        result = pipeline.ask(image, question, top_k=args.top_k)
        evidence_lines = [
            f"[{i}] {ev.title} | score={ev.score:.3f} | {ev.source}\n{ev.content}"
            for i, ev in enumerate(result.evidences, start=1)
        ]
        return result.answer, result.visual_caption, "\n\n".join(evidence_lines)

    demo = gr.Interface(
        fn=infer,
        inputs=[gr.Image(type="filepath", label="图像"), gr.Textbox(label="问题")],
        outputs=[gr.Textbox(label="答案"), gr.Textbox(label="图像描述"), gr.Textbox(label="支撑证据")],
        title="基于 RAG 的图像问答",
    )
    demo.launch(server_name=args.host, server_port=args.port)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-based visual question answering")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build local vector index")
    p_build.add_argument("--kb", default="data/knowledge_base/sample_knowledge.jsonl")
    p_build.add_argument("--index-dir", default="outputs/index")
    p_build.add_argument("--debug", action="store_true", help="Print intermediate variables to stderr")
    p_build.set_defaults(func=build_index)

    p_ask = sub.add_parser("ask", help="Ask a question about an image")
    p_ask.add_argument("--image", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--kb", default="data/knowledge_base/sample_knowledge.jsonl")
    p_ask.add_argument("--index-dir", default="outputs/index")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--web", action="store_true", help="Enable Wikipedia evidence retrieval")
    p_ask.add_argument("--debug", action="store_true", help="Print intermediate variables to stderr")
    p_ask.set_defaults(func=ask)

    p_eval_retrieval = sub.add_parser(
        "eval-retrieval", help="Compare baseline vs joint-query retrieval coverage"
    )
    p_eval_retrieval.add_argument("--eval", default="data/eval/retrieval_eval.jsonl")
    p_eval_retrieval.add_argument("--kb", default="data/knowledge_base/sample_knowledge.jsonl")
    p_eval_retrieval.add_argument("--index-dir", default="outputs/index")
    p_eval_retrieval.add_argument(
        "--output", default="outputs/retrieval_eval.json", help="Where to write the per-sample report"
    )
    p_eval_retrieval.add_argument(
        "--use-blip-caption",
        action="store_true",
        help="Run BLIP for captions when an eval sample has an image but no synthetic_caption",
    )
    p_eval_retrieval.add_argument("--debug", action="store_true")
    p_eval_retrieval.set_defaults(func=eval_retrieval)

    p_eval_okvqa = sub.add_parser("eval-okvqa", help="Score the pipeline on OKVQA val")
    p_eval_okvqa.add_argument("--questions", required=True, help="OKVQA OpenEnded questions JSON")
    p_eval_okvqa.add_argument("--annotations", required=True, help="OKVQA annotations JSON")
    p_eval_okvqa.add_argument("--images-dir", required=True, help="Directory containing COCO val2014 images")
    p_eval_okvqa.add_argument("--kb", default="data/knowledge_base/okvqa_kb.jsonl")
    p_eval_okvqa.add_argument("--index-dir", default="outputs/okvqa_index")
    p_eval_okvqa.add_argument("--output", default="outputs/okvqa_eval.json")
    p_eval_okvqa.add_argument("--top-k", type=int, default=5)
    p_eval_okvqa.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    p_eval_okvqa.add_argument("--web", action="store_true", help="Enable Wikipedia retrieval (slow)")
    p_eval_okvqa.add_argument("--cache-caption", action="store_true", help="Cache BLIP captions to disk")
    p_eval_okvqa.add_argument("--caption-cache", default="outputs/caption_cache.json")
    p_eval_okvqa.add_argument("--debug", action="store_true")
    p_eval_okvqa.set_defaults(func=eval_okvqa)

    p_serve = sub.add_parser("serve", help="Run a Gradio demo")
    p_serve.add_argument("--kb", default="data/knowledge_base/sample_knowledge.jsonl")
    p_serve.add_argument("--index-dir", default="outputs/index")
    p_serve.add_argument("--top-k", type=int, default=5)
    p_serve.add_argument("--web", action="store_true")
    p_serve.add_argument("--debug", action="store_true", help="Print intermediate variables to stderr")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=7860)
    p_serve.set_defaults(func=serve)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
