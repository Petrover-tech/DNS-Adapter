# DNS-Adapter

DNS-Adapter is a training-free dynamic neuro-symbolic framework for vocabulary-free few-shot fine-grained recognition. It keeps the original visual-only vocabulary-free projection pipeline, then selectively invokes external-knowledge reasoning for uncertain samples.

> This repository is a public scaffold for the DNS-Adapter paper. Key logic files are marked below and will be released after the paper is published.

## Method

DNS-Adapter first maps query images to anonymous candidate logits through generic ImageNet/WordNet anchors and a closed-form support-set mapping. For high-entropy predictions, it builds a local knowledge context, prompts an LLM to compare only the top visual candidates, applies trie-style candidate rectification, and fuses the symbolic residual back into the visual logits.

![DNS-Adapter overview](figures/dns_adapter_overview.jpg)

![Case study](figures/dns_adapter_case_study.jpg) | ![Symbolic reasoning](figures/dns_adapter_symbolic_reasoning.jpg) | ![Trie rectification](figures/dns_adapter_trie.jpg)

## Repository

- `main.py`: experiment entry point; preserves dataset loading, feature caching, support sampling, and closed-form visual mapping.
- `dns_adapter/visual_projection.py`: public vocabulary-free visual projection.
- `dns_adapter/knowledge_graph.py`: public local knowledge-graph scaffold.
- `dns_adapter/prompts.py`: public DNS-Adapter prompt templates and output parsing.
- `dns_adapter/trie.py`: public candidate trie interface.
- `dns_adapter/symbolic_reasoning.py`: public symbolic reasoning interface with a deterministic fallback.
- `dns_adapter/unreleased.py`: publication-pending placeholders for the final LLM scoring, calibration, trie-constrained decoding, and retrieval policies.
- `PROJECT_TREE.md`: concise project tree.

Key logic to be fully opened after publication: The whole `dns_adapter` file and `scripts file`.

## Setup

```bash
conda create -y --name dns_adapter python=3.10
conda activate dns_adapter
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

Install datasets following [DATASETS.md](DATASETS.md).

## Run

```bash
python main.py --dataset dtd --root_path /path/to/datasets --backbone vit_b16 --n_shots 16 --run_mode dns_adapter
```

Optional LLM scoring uses `DNS_ADAPTER_API_KEY` or `OPENAI_API_KEY`:

```bash
DNS_ADAPTER_API_KEY=... python main.py --dataset stanford_cars --root_path /path/to/datasets --run_mode dns_adapter
```

Batch scripts are provided in `scripts/`:

```bash
bash scripts/DNS_Adapter_imagenet_text.sh --root /path/to/datasets --backbone vit_b16 --n_shots 16
bash scripts/DNS_Adapter_wordnet.sh --root /path/to/datasets --backbone vit_b16 --n_shots 16
```

## Citation

Citation information will be added after the DNS-Adapter paper is published.

## License

This repository keeps the original AGPL-3.0 license.
