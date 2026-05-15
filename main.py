import argparse
import os
import random
import sys
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

import clip
import datasets as dts
import utils as uti
from dns_adapter import (
    DNSAdapterConfig,
    DNSAdapterReasoner,
    DomainKnowledgeGraph,
    compute_visual_logits,
    fit_visual_mapping,
    fuse_logits,
)
from dns_adapter.visual_projection import normalized_entropy

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_arguments():
    parser = argparse.ArgumentParser("DNS-Adapter")
    parser.add_argument("--dataset", default="dtd", type=str)
    parser.add_argument("--root_path", default="./datasets", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--source_prompts_types", default="imagenet_text", choices=["imagenet_text", "wordnet"])
    parser.add_argument("--method", default="DNS-Adapter", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--backbone", default="vit_b16", type=str)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--n_shots", type=int, default=16)
    parser.add_argument("--n_random_seeds", type=int, default=3)
    parser.add_argument("--augment_epoch", type=int, default=0)

    parser.add_argument("--run_mode", default="dns_adapter", choices=["visual_only", "dns_adapter"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--entropy_threshold", type=float, default=0.15)
    parser.add_argument("--gate_sharpness", type=float, default=10.0)
    parser.add_argument("--top_k_candidates", type=int, default=8)
    parser.add_argument("--symbolic_temperature", type=float, default=0.7)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--llm_sample_size", type=int, default=-1)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_shots(args, base_rng, train_labels, train_features):
    shots_rng = np.random.default_rng(base_rng.integers(0, 10000))
    n_shots = args.n_shots
    num_classes = int(torch.max(train_labels) + 1)
    shots_features = torch.zeros((n_shots * num_classes, *train_features.shape[1:]), dtype=train_features.dtype)
    shots_labels = torch.zeros((n_shots * num_classes), dtype=train_labels.dtype)

    train_features_cpu = train_features.cpu()
    train_labels_cpu = train_labels.cpu()
    for class_idx in range(num_classes):
        mask = train_labels_cpu == class_idx
        count = torch.sum(mask).item()
        selected = shots_rng.choice(count, n_shots, replace=count < n_shots)
        start = class_idx * n_shots
        end = (class_idx + 1) * n_shots
        shots_features[start:end] = train_features_cpu[mask][selected]
        shots_labels[start:end] = class_idx
    return shots_features, shots_labels


def load_source_prototypes(args, clip_model):
    if args.source_prompts_types == "imagenet_text":
        classnames = dts.imagenet.imagenet_classes
        prototypes = uti.clip_classifier(
            classnames,
            dts.imagenet.imagenet_templates,
            clip_model,
            reduce="mean",
        )
        return prototypes, classnames

    prototypes = uti.get_wordnet_prompts(args)
    wordnet_path = os.path.join(args.root_path, "filtered_wordnet_words.pickle")
    import pickle

    with open(wordnet_path, "rb") as f:
        classnames = pickle.load(f)
    return prototypes, classnames


def build_anchor_descriptions(features, source_prototypes, source_classnames, top_n=10) -> List[str]:
    source_prototypes = source_prototypes.to(features.device).float().squeeze()
    sims = 100.0 * features.float() @ source_prototypes
    k = min(top_n, sims.shape[-1])
    _, top_attrs = torch.topk(sims, k=k, dim=-1)
    descriptions = []
    for row in top_attrs.cpu().tolist():
        anchors = [str(source_classnames[idx]).replace("_", " ") for idx in row]
        descriptions.append("observable generic-anchor evidence: " + ", ".join(anchors))
    return descriptions


def make_llm_client(args) -> Optional[Callable]:
    api_key = args.api_key or os.environ.get("DNS_ADAPTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    if OpenAI is None:
        raise RuntimeError("Install openai to enable DNS-Adapter LLM scoring.")

    client = OpenAI(api_key=api_key, base_url=args.api_base)

    def complete(messages):
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=256,
        )
        return response.choices[0].message.content

    return complete


def evaluate_dns_adapter(args, mapping, dataset, test_features, test_labels, source_prototypes, source_classnames):
    test_features = test_features.to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    visual_logits = compute_visual_logits(test_features, source_prototypes.to(DEVICE), mapping.to(DEVICE))
    visual_preds = torch.argmax(visual_logits, dim=-1)
    visual_acc = (visual_preds == test_labels).float().mean().item()

    if args.run_mode == "visual_only":
        return visual_acc, visual_acc, 0

    entropy = normalized_entropy(visual_logits)
    graph = DomainKnowledgeGraph.from_generic_anchors(args.dataset, source_classnames)
    descriptions = build_anchor_descriptions(test_features, source_prototypes, source_classnames)
    config = DNSAdapterConfig(
        top_k_candidates=args.top_k_candidates,
        entropy_threshold=args.entropy_threshold,
        gate_sharpness=args.gate_sharpness,
        symbolic_temperature=args.symbolic_temperature,
        residual_scale=args.residual_scale,
        max_symbolic_samples=args.llm_sample_size,
    )
    reasoner = DNSAdapterReasoner(
        config=config,
        graph=graph,
        classnames=[str(name) for name in dataset.classnames],
        llm_client=make_llm_client(args),
    )
    symbolic = reasoner.build_symbolic_delta(visual_logits, descriptions, entropy)
    final_logits = fuse_logits(
        visual_logits,
        symbolic.delta,
        entropy,
        threshold=args.entropy_threshold,
        sharpness=args.gate_sharpness,
        residual_scale=args.residual_scale,
    )
    final_preds = torch.argmax(final_logits, dim=-1)
    final_acc = (final_preds == test_labels).float().mean().item()
    return visual_acc, final_acc, len(symbolic.routed_indices)


def main():
    args = get_arguments()
    set_random_seed(args.seed)
    backbones = {"rn50": "RN50", "rn101": "RN101", "vit_b16": "ViT-B/16", "vit_b32": "ViT-B/32", "vit_l14": "ViT-L/14"}

    try:
        clip_model, preprocess = clip.load(backbones[args.backbone], device=DEVICE)
        clip_model.eval()

        dataset_dirs = {
            "imagenet": "imagenet",
            "sun397": "sun397",
            "fgvc_aircraft": "fgvc_aircraft",
            "eurosat": "eurosat",
            "stanford_cars": "StanfordCars",
            "food101": "Food101",
            "oxford_pets": "OxfordPets",
            "oxford_flowers": "Flower102",
            "caltech101": "Caltech101",
            "dtd": "dtd",
            "ucf101": "UCF101",
        }
        cache_base = args.cache_dir or os.path.join(args.root_path, dataset_dirs.get(args.dataset, args.dataset), "cache")
        args.cache_dir = cache_base
        os.makedirs(args.cache_dir, exist_ok=True)

        _, _, _, dataset, features = uti.load_features(
            args.dataset,
            args.root_path,
            args.cache_dir,
            preprocess,
            clip_model,
            backbones[args.backbone],
            load_loaders=False,
        )
        train_feat, train_lab, test_feat, test_lab = features
        train_feat, train_lab = train_feat.to(DEVICE), train_lab.to(DEVICE)
        test_feat, test_lab = test_feat.to(DEVICE), test_lab.to(DEVICE)

        source_prototypes, source_classnames = load_source_prototypes(args, clip_model)
        source_prototypes = source_prototypes.to(DEVICE)
        lambda_reg = 0.2 if args.source_prompts_types == "wordnet" else 0.1

        visual_accs, dns_accs, routed_counts = [], [], []
        print(f"\n========== Running DNS-Adapter on {args.dataset} ==========")
        for seed_offset in tqdm(range(args.n_random_seeds)):
            set_random_seed(args.seed + seed_offset)
            rng = np.random.default_rng(args.seed + seed_offset)
            shots_feat, shots_lab = select_shots(args, rng, train_lab.cpu(), train_feat.cpu())
            mapping = fit_visual_mapping(
                shots_feat.to(DEVICE),
                shots_lab.to(DEVICE),
                source_prototypes,
                num_classes=len(dataset.classnames),
                lambda_reg=lambda_reg,
            )
            visual_acc, dns_acc, routed = evaluate_dns_adapter(
                args,
                mapping,
                dataset,
                test_feat,
                test_lab,
                source_prototypes,
                source_classnames,
            )
            visual_accs.append(visual_acc)
            dns_accs.append(dns_acc)
            routed_counts.append(routed)
            print(f"Seed {seed_offset}: visual={visual_acc:.2%}, dns={dns_acc:.2%}, routed={routed}")

        print(f"Visual Projection Accuracy: {np.mean(visual_accs):.4f}")
        print(f"DNS-Adapter Accuracy:       {np.mean(dns_accs):.4f}")
        print(f"Avg Routed Samples:         {np.mean(routed_counts):.1f}")
    except Exception as exc:
        print(f"CRITICAL: Unhandled exception: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
