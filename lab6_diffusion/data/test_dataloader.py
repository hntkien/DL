"""
Smoke test for dataset.py — run this after get_dataset.sh to verify loading.

Usage:
    python test_dataset.py \
        --image_dir  data/iclevr \
        --train_json data/train.json \
        --test_json  data/test.json \
        --new_test_json data/new_test.json \
        --objects_json  data/objects.json
"""
import argparse 
import torch 
from dataset import get_train_loader, get_test_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--new_test_json", type=str, required=True)
    parser.add_argument("--objects_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def test_train_loader(args: argparse.Namespace):
    print("=== Training DataLoader ===\n")
    loader = get_train_loader(
        train_json=args.train_json,
        image_dir=args.image_dir,
        objects_json=args.objects_json,
        batch_size=args.batch_size,
        drop_prob=0.1,
        num_workers=0,  # Use 0 workers for testing to avoid multiprocessing issues in notebooks
    )
    print(f" Dataset Size: {len(loader.dataset)}")
    print(f" Num. Batches: {len(loader)}")

    images, conditions = next(iter(loader))
    print(f" Image Batch: {tuple(images.shape)} (dtype={images.dtype})")
    print(f" Condition Batch: {tuple(conditions.shape)} (dtype={conditions.dtype})")
    print(f" Image Range: [{images.min(): .3f}, {images.max(): .3f}] (expect [-1, 1])")
    print(f" Condition Sum (per sample): {conditions.sum(dim=-1).tolist()}") 

    # Verify CFG null vectors appear sometimes 
    from dataset import ICLEVRTrainDataset
    ds = ICLEVRTrainDataset(
        train_json=args.train_json,
        image_dir=args.image_dir,
        objects_json=args.objects_json,
        image_size=64,
        drop_prob=1.0,  # Force all conditions to be dropped
    )
    _, cond = ds[0] 
    assert cond.sum().item() == 0.0, "CFG null vector should be all zeros"
    print(" CFG null-drop: OK (all zeros when drop_prob=1.0)")

def test_test_loader(args: argparse.Namespace, split_name: str, path: str):
    print(f"\n=== Test DataLoader ({split_name}) ===\n")
    loader = get_test_loader(
        test_json=path,
        objects_json=args.objects_json,
        batch_size=args.batch_size,
    )
    print(f" Dataset Size: {len(loader.dataset)}")
    print(f" Num. Batches: {len(loader)}")

    conditions = next(iter(loader))
    print(f" Condition Batch: {tuple(conditions.shape)} (dtype={conditions.dtype})")
    print(f" Condition Sum (per sample): {conditions.sum(dim=-1).tolist()}") 

def test_label_encoding(args: argparse.Namespace):
    print("\n=== Label Encoding ===\n")
    from utils import load_label_map, encode_labels
    label_map = load_label_map(args.objects_json)
    vec = encode_labels(["cyan cube", "red sphere"], label_map)
    assert vec.shape == (24,)
    assert vec[label_map["cyan cube"]] == 1.0
    assert vec[label_map["red sphere"]] == 1.0
    assert vec.sum().item() == 2.0
    print(
        f"  encode_labels(['cyan cube', 'red sphere']) -> sum={vec.sum().item()} at indices "
        f"{vec.nonzero(as_tuple=True)[0].tolist()}  OK"
    )

if __name__ == "__main__":
    args = parse_args()
    test_label_encoding(args)
    test_train_loader(args)
    test_test_loader(args, split_name="test.json", path=args.test_json)
    test_test_loader(args, split_name="new_test.json", path=args.new_test_json)
    print("\n[+] All checks passed.")