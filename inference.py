#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from transformers import (
    AutoImageProcessor,
    EfficientNetForImageClassification,
    ViTForImageClassification,
)

# The 5 target labels
TARGET_CLASSES = ["motorcycle", "truck", "boat", "bus", "cycle"]

# Model mapping: choice → (model_name, checkpoint_filename)
MODEL_SPECS = {
    "1": ("efficientnetb0",             "efficientnetb0_model.pt"),
    "2": ("dinov2-small",               "dinov2-small.pt"),
    "3": ("deit-tiny",                  "deit-tiny-patch16-224.pt"),
    "4": ("resnet18",                   "resnet18_classification.pt"),
}

# Will hold entries for comparison table
comparison_entries = []

def get_transform(model_name):
    if model_name == "efficientnetb0":
        size = 224
        proc = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
        mean, std = proc.image_mean, proc.image_std
    elif model_name == "dinov2-small":
        size = 518
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        mean, std = proc.image_mean, proc.image_std
    elif model_name == "deit-tiny":
        size = 224
        proc = AutoImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")
        mean, std = proc.image_mean, proc.image_std
    elif model_name == "resnet18":
        size = 224
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def load_model(model_name, checkpoint, device):
    if model_name == "efficientnetb0":
        mdl = EfficientNetForImageClassification.from_pretrained(
            "google/efficientnet-b0", num_labels=len(TARGET_CLASSES), ignore_mismatched_sizes=True)
    elif model_name == "dinov2-small":
        mdl = ViTForImageClassification.from_pretrained(
            "facebook/dinov2-small", num_labels=len(TARGET_CLASSES), ignore_mismatched_sizes=True)
    elif model_name == "deit-tiny":
        mdl = ViTForImageClassification.from_pretrained(
            "facebook/deit-tiny-patch16-224", num_labels=len(TARGET_CLASSES), ignore_mismatched_sizes=True)
    elif model_name == "resnet18":
        mdl = models.resnet18(pretrained=True)
        mdl.fc = torch.nn.Linear(mdl.fc.in_features, len(TARGET_CLASSES))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    state = torch.load(checkpoint, map_location=device)
    mdl.load_state_dict(state)
    mdl.to(device)
    mdl.eval()
    return mdl

def predict(mdl, model_name, img_tensor, device, thresh=0.5):
    img_tensor = img_tensor.to(device).unsqueeze(0)
    with torch.no_grad():
        if model_name == "resnet18":
            logits = mdl(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            return [(TARGET_CLASSES[idx], probs[idx])]
        else:
            out    = mdl(img_tensor)
            logits = out.logits if hasattr(out, "logits") else out
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
            results = [(c, float(probs[i])) for i, c in enumerate(TARGET_CLASSES) if probs[i] >= thresh]
            if not results:
                top3 = np.argsort(probs)[-3:][::-1]
                results = [(TARGET_CLASSES[i], float(probs[i])) for i in top3]
            return results

def run_once(model_key, image_num):
    model_name, checkpoint = MODEL_SPECS[model_key]
    img_file = f"{image_num}.jpg"
    if not os.path.exists(checkpoint):
        print(f"[ERROR] Missing checkpoint: {checkpoint}")
        return
    if not os.path.exists(img_file):
        print(f"[ERROR] Missing image: {img_file}")
        return

    device = torch.device("cpu")
    transform = get_transform(model_name)
    mdl = load_model(model_name, checkpoint, device)
    img = Image.open(img_file).convert("RGB")
    img_tensor = transform(img)
    results = predict(mdl, model_name, img_tensor, device)

    summary = "; ".join(f"{lbl}({pr:.2f})" for lbl, pr in results)
    comparison_entries.append({
        "Image": img_file,
        "Model": model_name,
        "Prediction": summary
    })

    print(f"\nModel: {model_name} | Image: {img_file}")
    if model_name == "resnet18":
        lbl, pr = results[0]
        print(f"→ {lbl} (prob={pr:.4f})")
    else:
        print("→ Labels:")
        for lbl, pr in results:
            print(f"   • {lbl}: {pr:.4f}")

def auto_mode():
    print("\nAUTO MODE: running all models on images 1–10...\n")
    for mk in MODEL_SPECS:
        print(f"--- {MODEL_SPECS[mk][0].upper()} ---")
        for i in range(1, 11):
            run_once(mk, i)

    # Build comparison table
    df = pd.DataFrame(comparison_entries)
    table = df.pivot(index="Image", columns="Model", values="Prediction").fillna("-")

    print("\n=== COMPARISON TABLE ===")
    print(table.to_string())

    with open("model_comparison_output.txt", "w") as f:
        f.write(table.to_string())

    input("\nAUTO MODE complete. Press Enter to exit...")

def manual_mode():
    print("\nMANUAL MODE: select one model and one image.\n")
    print("Select model:")
    for k, (mn, _) in MODEL_SPECS.items():
        print(f"  {k}. {mn}")
    mk = input("Enter model number (1–4): ").strip()
    if mk not in MODEL_SPECS:
        print("Invalid model choice."); input("Press Enter to exit..."); return

    inum = input("Enter image number (1–10): ").strip()
    if not inum.isdigit() or not (1 <= int(inum) <= 10):
        print("Invalid image number."); input("Press Enter to exit..."); return

    run_once(mk, inum)
    input("\nManual MODE complete. Press Enter to exit...")

def main():
    print("Choose mode:")
    print("  1. AUTO MODE   – run all models on images 1–10 and output comparison table")
    print("  2. MANUAL MODE – select specific model and image")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        auto_mode()
    elif choice == "2":
        manual_mode()
    else:
        print("Invalid choice."); input("Press Enter to exit...")

if __name__ == "__main__":
    import pandas as pd
    main()
