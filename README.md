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

<table>
  <tr>
    <td align="center" width="33%">
      <img src="figures/dns_adapter_case_study.jpg" alt="Case study" width="100%">
      <br>
      <b>Case study</b>
    </td>
    <td align="center" width="33%">
      <img src="figures/dns_adapter_symbolic_reasoning.jpg" alt="Symbolic reasoning" width="100%">
      <br>
      <b>Symbolic reasoning</b>
    </td>
    <td align="center" width="33%">
      <img src="figures/dns_adapter_trie.jpg" alt="Trie rectification" width="100%">
      <br>
      <b>Trie rectification</b>
    </td>
  </tr>
</table>

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
