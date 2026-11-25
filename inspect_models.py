from pathlib import Path

import torch
from ultralytics import YOLO


MODELS = [
    Path("models/temp_best_ball.pt"),
    Path("models/best_resnet_whistle_mix.pth"),
    Path("models/rally_model_best.pth"),
]


def inspect_yolo(path):
    try:
        model = YOLO(str(path))

        print("Type: Ultralytics YOLO")
        print("Task:", model.task)
        print("Classes:", model.names)

        inner = model.model
        print("Architecture:", type(inner).__name__)

        args = getattr(inner, "args", {})

        if args:
            print("\nTraining/model arguments:")

            keys = [
                "model",
                "data",
                "imgsz",
                "batch",
                "epochs",
                "task",
                "optimizer",
            ]

            for key in keys:
                if key in args:
                    print(f"  {key}: {args[key]}")

        return True

    except Exception as e:
        print(f"Could not load as Ultralytics YOLO: {e}")
        return False


def inspect_pytorch(path):
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        print("Checkpoint type:", type(checkpoint).__name__)

        if not isinstance(checkpoint, dict):
            print(checkpoint)
            return

        print("Checkpoint keys:", list(checkpoint.keys()))

        metadata_keys = [
            "epoch",
            "model_name",
            "architecture",
            "model_type",
            "best_val_f1",
            "val_f1",
            "args",
            "config",
        ]

        print("\nMetadata:")

        found_metadata = False

        for key in metadata_keys:
            if key in checkpoint:
                print(f"  {key}: {checkpoint[key]}")
                found_metadata = True

        if not found_metadata:
            print("  No known metadata fields found.")

        state = checkpoint.get(
            "model_state",
            checkpoint.get("state_dict"),
        )

        # Some checkpoints are themselves a raw state_dict
        if state is None:
            if all(
                isinstance(value, torch.Tensor)
                for value in checkpoint.values()
            ):
                state = checkpoint

        if state is not None:
            print(f"\nState dict entries: {len(state)}")
            print("First layers:")

            for name, tensor in list(state.items())[:15]:
                if hasattr(tensor, "shape"):
                    print(f"  {name}: {tuple(tensor.shape)}")

    except Exception as e:
        print(f"Could not inspect PyTorch checkpoint: {e}")


def inspect_model(path):
    print("\n" + "=" * 80)
    print(f"MODEL: {path}")
    print("=" * 80)

    if not path.exists():
        print("MISSING")
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

    if path.suffix == ".pt":
        if inspect_yolo(path):
            return

    inspect_pytorch(path)


def main():
    for model_path in MODELS:
        inspect_model(model_path)


if __name__ == "__main__":
    main()
