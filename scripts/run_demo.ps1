param(
    [string]$Image = "examples/eiffel_demo.jpg",
    [string]$Question = "What is the historical significance of this iron tower?"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path "outputs/index/documents.json")) {
    Write-Host ">>> Building knowledge-base index"
    python -m rag_vqa.cli build-index --kb data/knowledge_base/sample_knowledge.jsonl --index-dir outputs/index
}

Write-Host ">>> Asking the pipeline (text + image + web evidence + interpretable answer)"
python -m rag_vqa.cli ask `
    --image $Image `
    --question $Question `
    --kb data/knowledge_base/sample_knowledge.jsonl `
    --index-dir outputs/index `
    --top-k 5
