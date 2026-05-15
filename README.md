# DNS-Adapter

DNS-Adapter is a training-free dynamic neuro-symbolic framework for vocabulary-free few-shot fine-grained recognition. It keeps the original visual-only vocabulary-free projection pipeline, then selectively invokes external-knowledge reasoning for uncertain samples.

> This repository is a public scaffold for the DNS-Adapter paper. Key logic files are marked below and will be released after the paper is published.

## Method

DNS-Adapter first maps query images to anonymous candidate logits through generic ImageNet/WordNet anchors and a closed-form support-set mapping. For high-entropy predictions, it builds a local knowledge context, prompts an LLM to compare only the top visual candidates, applies trie-style candidate rectification, and fuses the symbolic residual back into the visual logits.

<p align="center">
  <img src="figures/dns_adapter_overview.jpg" alt="DNS-Adapter overview" width="720">
</p>

<p align="center">
  <b>Overview of DNS-Adapter</b>
</p>

<p align="center">
  <img src="figures/dns_adapter_case_study.jpg" alt="Case study" width="720">
</p>

<p align="center">
  <b>Case study</b>
</p>

<p align="center">
  <img src="figures/dns_adapter_symbolic_reasoning.jpg" alt="Symbolic reasoning" width="720">
</p>

<p align="center">
  <b>Symbolic reasoning</b>
</p>

<p align="center">
  <img src="figures/dns_adapter_trie.jpg" alt="Trie rectification" width="720">
</p>

<p align="center">
  <b>Trie rectification</b>
</p>

## Repository

The repository is organized around the DNS-Adapter pipeline: dataset preparation, vocabulary-free visual projection, uncertainty-aware symbolic reasoning, and result reproduction.

- `main.py`: main experiment entry point. It keeps the original data-processing path, including dataset loading, feature caching, support-set sampling, generic-anchor similarity computation, and closed-form visual mapping. It then calls the DNS-Adapter modules for entropy-based routing, symbolic correction, and residual fusion.

- `dns_adapter/`: core DNS-Adapter package. This folder contains the public modular implementation of the method described in the paper. The files are separated by function so that the visual branch, knowledge branch, prompt branch, constrained rectification, and fusion logic can be inspected independently.

- `dns_adapter/visual_projection.py`: vocabulary-free visual projection module. It fits the closed-form mapping from generic-anchor similarities to anonymous support labels and computes visual logits for query images. This is the preserved visual backbone of the project.

- `dns_adapter/fusion.py`: uncertainty-weighted residual fusion module. It implements the routing weight used to decide how strongly symbolic evidence should modify visual logits.

- `dns_adapter/config.py`: DNS-Adapter runtime configuration and dataset-domain profiles. It stores default thresholds, top-k settings, symbolic temperature, and concise domain descriptions used by the prompt builder.

- `dns_adapter/knowledge_graph.py`: public local knowledge-graph scaffold. It provides the interface for constructing candidate-level external knowledge and retrieving local subgraphs for ambiguous samples.

- `dns_adapter/prompts.py`: DNS-Adapter prompt construction and LLM output parsing. It builds image-description and candidate-restricted reasoning prompts, then parses symbolic scores from JSON responses.

- `dns_adapter/trie.py`: candidate trie interface. It represents the trie-constrained rectification idea from the paper and keeps symbolic outputs inside the valid top-k candidate space.

- `dns_adapter/symbolic_reasoning.py`: symbolic reasoning controller. It selects routed samples, retrieves candidate evidence, builds symbolic scores, and returns a residual correction vector. The public version includes a deterministic fallback so the repository remains runnable without private LLM logic.

- `dns_adapter/unreleased.py`: publication-pending placeholders. This file explicitly marks the final paper-specific LLM scoring, calibration, trie-constrained decoding, and external-knowledge retrieval policies that will be released after publication.

- `datasets/`: dataset wrappers and split loaders. This folder is retained from the original experiment code so DNS-Adapter can use the same benchmarks, split files, class-index labels, and dataloader behavior.

- `scripts/`: convenience scripts for reproducing multi-dataset runs. The script names have been updated to DNS-Adapter and call `main.py` with ImageNet-text or WordNet generic anchors.

- `figures/`: README and paper illustration assets. It contains DNS-Adapter figures copied from the paper package, including the overview, symbolic reasoning branch, trie rectification, and case study.

- `auto_generate_prompt.py`: optional helper for generating or printing domain prompt profiles. It reads API keys only from environment variables and can also run in offline mode.

- `DATASETS.md`: dataset installation and layout guide. It explains the expected root directory structure and notes how DNS-Adapter uses class indices while avoiding target textual anchors.

- `PROJECT_TREE.md`: concise project tree. It gives a quick structural overview of all important files and folders.

Key logic to be fully opened after publication: The `dns_adapter` file and `scripts` file.

## Setup

```bash
conda create -y --name dns_adapter python=3.10
conda activate dns_adapter
pip install -r requirements.txt
pip install torch torchvision torchaudio
