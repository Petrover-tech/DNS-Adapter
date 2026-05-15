# Dataset Setup for DNS-Adapter

All datasets should be placed under one root folder, passed as `--root_path`. DNS-Adapter preserves the original data-processing logic and expects the same dataset names and split files.

```
$DATA/
|-- Caltech101/
|-- OxfordPets/
|-- StanfordCars/
|-- Flower102/
|-- Food101/
|-- fgvc_aircraft/
|-- sun397/
|-- dtd/
|-- eurosat/
|-- UCF101/
|-- imagenet/
```

For the standard few-shot splits, use the Zhou et al. split JSON files commonly used by CLIP adaptation methods. DNS-Adapter uses class indices for support labels during visual mapping; fine-grained class names are not used to construct target textual anchors.

## Required Datasets

- Caltech101: `Caltech101/101_ObjectCategories/` and `split_zhou_Caltech101.json`
- OxfordPets: `OxfordPets/images/`, `OxfordPets/annotations/`, and `split_zhou_OxfordPets.json`
- StanfordCars: `StanfordCars/cars_train/`, `StanfordCars/cars_test/`, metadata files, and `split_zhou_StanfordCars.json`
- Flowers102: `Flower102/jpg/`, `imagelabels.mat`, `cat_to_name.json`, and `split_zhou_OxfordFlowers.json`
- Food101: `Food101/images/`, `Food101/meta/`, and `split_zhou_Food101.json`
- FGVCAircraft: `fgvc_aircraft/images/` plus the aircraft metadata text files
- SUN397: `sun397/SUN397/`, partition files, and `split_zhou_SUN397.json`
- DTD: `dtd/images/`, `dtd/imdb/`, `dtd/labels/`, and `split_zhou_DescribableTextures.json`
- EuroSAT: `eurosat/2750/` and `split_zhou_EuroSAT.json`
- UCF101: `UCF101/UCF-101-midframes/` and `split_zhou_UCF101.json`
- ImageNet: `imagenet/images/train/`, `imagenet/images/val/`, and `classnames.txt`

## Knowledge Sources

For `--source_prompts_types imagenet_text`, DNS-Adapter uses ImageNet generic textual anchors included in the dataset module. For `--source_prompts_types wordnet`, the code builds or loads WordNet anchors under `$DATA/wordnet_data/` and caches textual features under `$DATA/`.

The final paper-specific knowledge retrieval and calibration recipe is represented by public interfaces in `dns_adapter/knowledge_graph.py` and `dns_adapter/unreleased.py`; the key logic will be released after publication.
