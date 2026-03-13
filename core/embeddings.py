import json
import math
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from core.config import get_config

# Global state for lazy loading
_clip_model = None
_clip_processor = None
_faiss_index = None
_metadata_map = {}  # Maps FAISS index pos to {element_id, x_pct, y_pct}
_device = None

def _get_index_path() -> Path:
    config = get_config()
    index_path = config.get("paths", {}).get("faiss_index", "./assets/faiss.index")
    # Resolve relative to project root (which config.py handles or just assume CWD)
    # The config.yaml paths are usually relative to project root.
    return Path(index_path)

def _get_meta_path() -> Path:
    index_path = _get_index_path()
    return index_path.with_name(index_path.name + ".meta.json")

def _load_model():
    global _clip_model, _clip_processor, _device
    if _clip_model is not None:
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor

    config = get_config()
    model_name = config.get("models", {}).get("clip", "openai/clip-vit-base-patch32")
    gpu_idx = config.get("hardware", {}).get("gpu_embeddings", 1)

    if torch.cuda.is_available() and gpu_idx >= 0 and gpu_idx < torch.cuda.device_count():
        _device = torch.device(f"cuda:{gpu_idx}")
    else:
        _device = torch.device("cpu")

    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model = CLIPModel.from_pretrained(model_name).to(_device)

def generate_embedding(img: np.ndarray) -> np.ndarray:
    """Generates a normalized 512-dim CLIP embedding from an image.
    
    Args:
        img: np.ndarray of shape (H, W, 3) in RGB format.
        
    Returns:
        np.ndarray of shape (1, 512) containing the L2-normalized embedding.
    """
    _load_model()
    import torch
    
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
        
    pil_img = Image.fromarray(img)
    
    inputs = _clip_processor(images=pil_img, return_tensors="pt").to(_device)
    
    with torch.no_grad():
        image_features = _clip_model.get_image_features(**inputs)
        
    # L2 normalize for cosine similarity via inner product
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype("float32")

def load_index() -> faiss.Index:
    """Loads FAISS index and metadata from disk, or creates new ones if missing."""
    global _faiss_index, _metadata_map
    
    if _faiss_index is not None:
        return _faiss_index

    index_path = _get_index_path()
    meta_path = _get_meta_path()

    if index_path.exists() and meta_path.exists():
        _faiss_index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            # Ensure keys are integers (JSON keys are strings)
            _metadata_map = {int(k): v for k, v in meta_data.items()}
    else:
        # Create new index (512 is standard CLIP vision output dim)
        _faiss_index = faiss.IndexFlatIP(512)
        _metadata_map = {}
        
    return _faiss_index

def save_index():
    """Saves FAISS index and metadata sidecar to disk."""
    global _faiss_index, _metadata_map
    if _faiss_index is None:
        return

    index_path = _get_index_path()
    meta_path = _get_meta_path()
    
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_faiss_index, str(index_path))
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_metadata_map, f, indent=2)

def save_to_index(element_id: str, embedding: np.ndarray, x_pct: float = 0.0, y_pct: float = 0.0):
    """Adds an embedding to the FAISS index with spatial metadata and saves."""
    global _faiss_index, _metadata_map
    load_index()
    
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
        
    # Ensure it's float32 and C-contiguous as expected by FAISS
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)

    idx = _faiss_index.ntotal
    _faiss_index.add(embedding)
    
    _metadata_map[idx] = {
        "element_id": element_id,
        "x_pct": x_pct,
        "y_pct": y_pct
    }
    
    save_index()

def get_embedding_by_id(element_id: str) -> np.ndarray | None:
    """Retrieves the saved CLIP embedding for a specific element.

    This is used during replay to get the ORIGINAL embedding from recording
    time, so we can compare it against current screen crops (correct CLIP
    direction: saved → current, not current → saved).

    Args:
        element_id: The node ID stored during recording.

    Returns:
        np.ndarray of shape (1, 512) or None if not found.
    """
    global _faiss_index, _metadata_map
    load_index()

    for idx, meta in _metadata_map.items():
        if meta["element_id"] == element_id:
            if _faiss_index is not None and idx < _faiss_index.ntotal:
                vec = np.zeros((1, 512), dtype="float32")
                _faiss_index.reconstruct(idx, vec[0])
                return vec
    return None


def search_index(
    query_embedding: np.ndarray,
    top_k: int = 5,
    radius_pct: float = 0.20,
    query_x_pct: float | None = None,
    query_y_pct: float | None = None
) -> list[dict]:
    """Searches index, optionally filtering by position radius.
    
    Args:
        query_embedding: The embedding to search for.
        top_k: Max results to return before filtering.
        radius_pct: Max allowed euclidean distance in pct coordinate space (0.0 to 1.414).
        query_x_pct: Optional x position to filter around.
        query_y_pct: Optional y position to filter around.
        
    Returns:
        List of dicts: {"element_id": str, "score": float, "x_pct": float, "y_pct": float}
    """
    global _faiss_index, _metadata_map
    load_index()
    
    if _faiss_index.ntotal == 0:
        return []
        
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        
    # Fetch more results if we're going to filter spatially
    search_k = min(max(top_k * 3, 20), _faiss_index.ntotal) if query_x_pct is not None and query_y_pct is not None else min(top_k, _faiss_index.ntotal)
    
    scores, indices = _faiss_index.search(query_embedding, search_k)
    
    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        score = float(scores[0][i])
        
        if idx == -1 or idx not in _metadata_map:
            continue
            
        meta = _metadata_map[idx]
        
        # Spatial filtering
        if query_x_pct is not None and query_y_pct is not None:
            dx = meta["x_pct"] - query_x_pct
            dy = meta["y_pct"] - query_y_pct
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > radius_pct:
                continue
                
        results.append({
            "element_id": meta["element_id"],
            "score": score,
            "x_pct": meta["x_pct"],
            "y_pct": meta["y_pct"]
        })
        
        if len(results) >= top_k:
            break
            
    return results