from __future__ import annotations

import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "features")
os.makedirs(CACHE, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_IMG = 20000


def _save(name, X, y):
    np.savez_compressed(os.path.join(CACHE, f"{name}.npz"),
                        X=X.astype(np.float32), y=y.astype(np.int64))
    print(f"  {name}: X{X.shape} alpha={y.mean():.3f}")


def cifar():
    import torchvision
    from torchvision import transforms
    root = os.path.join(CACHE, "cifar_raw")
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    imgs = tr.data[:N_IMG]
    labels = np.array(tr.targets[:N_IMG])
    y = np.isin(labels, [0, 1, 8, 9]).astype(int)

    if not os.path.exists(os.path.join(CACHE, "cifar_raw.npz")):
        _save("cifar_raw", imgs.reshape(len(imgs), -1) / 255.0, y)

    if os.path.exists(os.path.join(CACHE, "cifar_clip.npz")):
        return
    from transformers import CLIPModel, CLIPImageProcessor
    mid = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(mid).to(DEV).eval()
    proc = CLIPImageProcessor.from_pretrained(mid)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            b = torch.tensor(imgs[i:i + 256]).permute(0, 3, 1, 2).float() / 255.0
            b = torch.nn.functional.interpolate(b, size=224, mode="bilinear",
                                                align_corners=False)
            b = ((b - mean) / std).to(DEV)
            out = model.get_image_features(pixel_values=b)


            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(emb.cpu().numpy())
            if i % 5120 == 0:
                print(f"    clip {i}/{len(imgs)}", flush=True)
    _save("cifar_clip", np.vstack(feats), y)


def newsgroups():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = d.data
    names = np.array(d.target_names)[d.target]
    y = np.char.startswith(names.astype(str), "comp.").astype(int)

    if not os.path.exists(os.path.join(CACHE, "news_tfidf.npz")):
        tf = TfidfVectorizer(max_features=20000, stop_words="english")
        Z = tf.fit_transform(texts)
        _save("news_tfidf", TruncatedSVD(512, random_state=0).fit_transform(Z), y)

    if os.path.exists(os.path.join(CACHE, "news_minilm.npz")):
        return
    from transformers import AutoModel, AutoTokenizer
    mid = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid).to(DEV).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            enc = tok(texts[i:i + 128], padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(DEV)
            out = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            feats.append(((out * m).sum(1) / m.sum(1)).cpu().numpy())
            if i % 2560 == 0:
                print(f"    minilm {i}/{len(texts)}", flush=True)
    _save("news_minilm", np.vstack(feats), y)


if __name__ == "__main__":
    print(f"device: {DEV}")
    cifar()
    newsgroups()
    print("cached ->", CACHE)
