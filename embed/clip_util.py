"""CLIP(open_clip ViT-B-32, 512-dim) 이미지 임베딩 + Qdrant 정품 레퍼런스 인덱스(in-memory)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_model = _pre = None
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    global _model, _pre
    if _model is None:
        _model, _, _pre = __import__("open_clip").create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        _model = _model.to(_DEV).eval()
    return _model, _pre


def embed(img) -> np.ndarray:
    """path(str/Path) 또는 PIL.Image → L2 정규화 512-d 벡터."""
    m, pre = load()
    if isinstance(img, (str, Path)):
        img = Image.open(img).convert("RGB")
    x = pre(img.convert("RGB")).unsqueeze(0).to(_DEV)
    with torch.no_grad():
        v = m.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy()


class ReferenceIndex:
    """정품 임베딩(reference.npz)을 Qdrant in-memory에 적재 → 쿼리 유사도."""

    def __init__(self, npz_path):
        from qdrant_client import QdrantClient, models
        ref = np.load(npz_path)["vecs"].astype(float)
        self.client = QdrantClient(":memory:")
        self.models = models
        self.client.create_collection(
            "ref", vectors_config=models.VectorParams(size=ref.shape[1], distance=models.Distance.COSINE))
        self.client.upsert("ref", points=[
            models.PointStruct(id=i, vector=ref[i].tolist()) for i in range(len(ref))], wait=True)

    def genuine_similarity(self, img, k: int = 5) -> float:
        """정품 레퍼런스 top-k 평균 코사인 유사도 (높을수록 정품에 가까움)."""
        v = embed(img)
        hits = self.client.query_points("ref", query=v.tolist(), limit=k).points
        return float(np.mean([h.score for h in hits])) if hits else 0.0
