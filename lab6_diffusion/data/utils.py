import os 
import json 
from typing import List, Dict
import torch 

def load_label_map(objects_json: str | os.PathLike) -> Dict[str, int]:
    """Load the object-name-to-index mapping from objects.json.

    Args:
        objects_json: Path to objects.json.

    Returns:
        Dict mapping object name strings to integer indices (0-23).
    """
    with open(objects_json, "r") as f:
        return json.load(f)
    
def encode_labels(
        labels: List[str], 
        label_map: Dict[str, int], 
        num_classes: int = 24
) -> torch.Tensor:
    """Convert a list of object-name strings to a multi-hot float32 tensor. 

    Args:
        labels (List[str]): List of object name strings, e.g. ["cyan cube", "red sphere"].
        label_map (Dict[str, int]): Mapping from object names to indices.
        num_classes (int, optional): Number of classes. Defaults to 24.

    Returns:
        torch.Tensor: Multi-hot float32 tensor of shape (num_classes,).
    """
    vec = torch.zeros(num_classes, dtype=torch.float32) 
    for name in labels:
        vec[label_map[name]] = 1.0 
    return vec 

