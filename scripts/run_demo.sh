#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-examples/eiffel_demo.jpg}"
QUESTION="${2:-What is the historical significance of this iron tower?}"

cd "$(dirname "$0")/.."

if [ ! -f "outputs/index/documents.json" ]; then
    echo ">>> Building knowledge-base index"
    python -m rag_vqa.cli build-index \
        --kb data/knowledge_base/sample_knowledge.jsonl \
        --index-dir outputs/index
fi

echo ">>> Asking the pipeline (text + image + web evidence + interpretable answer)"
python -m rag_vqa.cli ask \
    --image "$IMAGE" \
    --question "$QUESTION" \
    --kb data/knowledge_base/sample_knowledge.jsonl \
    --index-dir outputs/index \
    --top-k 5
