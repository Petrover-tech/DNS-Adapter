from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import clip
import datasets as dts
import pickle
import nltk
from nltk.corpus import wordnet

# [Fix] 智能设备检测
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template, clip_model, reduce='mean', gpt=False, wordnet_dict=None):
    with torch.no_grad():
        clip_weights = []
        if wordnet_dict is not None:
            indices = []
            i = 0
            for classname in classnames:
                allnames = [classname] + wordnet_dict[classname]
                for name in allnames:
                    name = name.replace('_', ' ')
                    texts = [t.format(name) for t in template]
                    # [Fix] .cuda() -> .to(DEVICE)
                    texts = clip.tokenize(texts).to(DEVICE)
                    class_embeddings = clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if reduce=='mean':
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        clip_weights.append(class_embedding)
                    if reduce is None:
                        class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                        clip_weights.append(class_embeddings)
                    i+=1
                indices.append(i)
            return clip_weights, indices
        else:
            for classname in classnames:
                classname = classname.replace('_', ' ')
                if gpt:
                    texts = template[classname]
                else:
                    texts = [t.format(classname) for t in template]
                # [Fix] .cuda() -> .to(DEVICE)
                texts = clip.tokenize(texts).to(DEVICE)
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if reduce=='mean':
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    clip_weights.append(class_embedding)
                if reduce is None:
                    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                    clip_weights.append(class_embeddings)
            # [Fix] .cuda() -> .to(DEVICE)
            clip_weights = torch.stack(clip_weights, dim=-1).to(DEVICE)
    return clip_weights

def pre_load_features(args, split, clip_model, loader, n_views=1, backbone_name='ViT-B/16'):
    model_cache_name = backbone_name.replace('/', '_').replace('-', '_')
    
    # [Fix] 区分增强特征和普通特征的文件名，防止混淆
    if n_views > 1:
        feat_path = os.path.join(args.cache_dir, f'{model_cache_name}_{split}_aug_{n_views}_features.pt')
        lbl_path = os.path.join(args.cache_dir, f'{split}_aug_{n_views}_target.pt')
    else:
        feat_path = os.path.join(args.cache_dir, f'{model_cache_name}_{split}_features.pt')
        lbl_path = os.path.join(args.cache_dir, f'{split}_target.pt')

    if not args.load or not (os.path.exists(feat_path) and os.path.exists(lbl_path)):
        if args.load: print(f"Cache missing at {feat_path}, computing...")
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for view in range(n_views):
                iterator = tqdm(loader, desc=f"Extract {split} View {view+1}", leave=False)
                for i, (images, target) in enumerate(iterator):
                    # [Fix] .cuda() -> .to(DEVICE)
                    images, target = images.to(DEVICE), target.to(DEVICE)
                    image_features = clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    all_features.append(image_features.cpu())
                    all_labels.append(target.cpu())
                            
        # 拼接所有特征
        features = torch.cat(all_features)
        labels = torch.cat(all_labels)

        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save(features, feat_path)
        torch.save(labels, lbl_path)
    else:
        try:
            features = torch.load(feat_path)
            labels = torch.load(lbl_path)
        except Exception:
            args.load = False
            return pre_load_features(args, split, clip_model, loader, n_views, backbone_name)
    
    return features, labels

def get_samples_feature_and_labels(cache_dir, splits=['test'], backbone_name='ViT-B/16', dataset_name=''):
    if 'EVA' not in backbone_name:
        model_cache_name = backbone_name.replace('/', '_').replace('-', '_')
        out = []
        for spl in splits:
            features_path = os.path.join(cache_dir, f'{model_cache_name}_{spl}_features.pt')
            lbl_path = os.path.join(cache_dir, f'{spl}_target.pt')
            
            if not os.path.exists(features_path):
                raise FileNotFoundError(f'Cache missing: {features_path}')
            
            # [Fix] .cuda() -> .to(DEVICE)
            _features = torch.load(features_path).to(DEVICE)
            _labels = torch.load(lbl_path).to(DEVICE)
            out.append(_features)
            out.append(_labels)
    else:
        out = []
    return out

def load_features(dataset_name, root_path, cache_dir, preprocess, clip_model, backbone_name, splits=['train', 'test'], load_loaders=False):
    class Cfg: pass
    cfg = Cfg()
    cfg.dataset = dataset_name
    cfg.root_path = root_path
    cfg.shots = 16
    
    train_loader, val_loader, test_loader, dataset = dts.get_all_dataloaders(cfg, preprocess)
    
    features_list = []
    dummy_args = type('Args', (), {'cache_dir': cache_dir, 'load': True})()
    
    if 'train' in splits:
        f, l = pre_load_features(dummy_args, 'train', clip_model, train_loader, backbone_name=backbone_name)
        features_list.extend([f, l])
    if 'test' in splits:
        f, l = pre_load_features(dummy_args, 'test', clip_model, test_loader, backbone_name=backbone_name)
        features_list.extend([f, l])
        
    return train_loader, val_loader, test_loader, dataset, features_list

def get_wordnet_prompts(args):
    nltk_data_path = os.path.join(args.root_path, 'wordnet_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet', download_dir=nltk_data_path)

    all_words = set()
    root_terms = ['building', 'vehicle', 'food', 'flower', 'animal', 'texture', 'action', 'furniture']
    for r in root_terms:
        for synset in wordnet.synsets(r, pos='n'):
            li_words = synset.lemma_names()
            cleaned_li_words = [w.replace('_', ' ').replace('.', '') for w in li_words if not(w[0] in ['1','2','3','4','5','6','7','8','9'])]
            all_words.update(cleaned_li_words)
            for hyponym in synset.closure(lambda s: s.hyponyms()):
                li_words = [w.replace('_', ' ').replace('.', '') for w in hyponym.lemma_names()]
                all_words.update(li_words)
    all_words = sorted(all_words)
    
    with open(os.path.join(args.root_path, 'filtered_wordnet_words.pickle'), 'wb') as f:
        pickle.dump(all_words, f)
        
    backbones = {'vit_b16': 'ViT-B/16', 'rn50': 'RN50', 'vit_b32':'ViT-B/32', 'rn101': 'RN101', 'vit_l14':'ViT-L/14'}
    model_name = args.backbone
    try:
        with open(os.path.join(args.root_path, f'filtered_wordnet_textual_features_{model_name}.pickle'), 'rb') as f:
            all_textual_features = pickle.load(f)
    except FileNotFoundError:
        all_textual_features = None 
        clip_model, preprocess = clip.load(backbones[model_name], device=DEVICE) # [Fix] 加载到 DEVICE
        batch_size = 64
        current_start = 0
        texts = []
        for jw , w in enumerate(tqdm(all_words)):
            texts.append(f'A photo of a {w}.')
            if len(texts) == batch_size or jw == len(all_words)-1:
                with torch.no_grad():
                    # [Fix] .cuda() -> .to(DEVICE)
                    texts_token = clip.tokenize(texts).to(DEVICE)
                    current_end = current_start + len(texts)
                    class_embeddings = clip_model.encode_text(texts_token)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if all_textual_features is None:
                        all_textual_features = torch.zeros((len(all_words), class_embeddings.shape[-1]), dtype=torch.float16)
                    all_textual_features[current_start:current_end,:] = class_embeddings.cpu().half()
                    current_start = current_end
                    texts = []
        with open(os.path.join(args.root_path, f'filtered_wordnet_textual_features_{model_name}.pickle'), 'wb') as f:
            pickle.dump(all_textual_features, f)
    # [Fix] .cuda() -> .to(DEVICE)
    return all_textual_features.to(DEVICE).T
