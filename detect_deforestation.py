"""
Deforestation detection: compares two satellite images of the same area
(different years) by classifying a grid of patches with the trained
EuroSAT model and flagging cells that changed FROM Forest TO something else.

Run with:  python detect_deforestation.py

Expects in the same folder:
  - eurosat_resnet50.pth   (your trained model)
  - image_before.jpg       (rename your 2018 image to this)
  - image_after.jpg        (rename your 2025 image to this)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- Config ----------
BEFORE_IMG = "image_before.jpg"
AFTER_IMG = "image_after.jpg"
MODEL_PATH = "eurosat_resnet50.pth"
PATCH_SIZE = 130   # pixels per grid cell (roughly square patches across the image)
BEFORE_LABEL = "2018"
AFTER_LABEL = "2025"

CLASS_NAMES = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway',
               'Industrial Buildings', 'Pasture', 'Permanent Crop',
               'Residential Buildings', 'River', 'SeaLake']

# Distinct colors for the change map
CLASS_COLORS = {
    'Annual Crop': '#e8d28a', 'Forest': '#1b5e20', 'Herbaceous Vegetation': '#8bc34a',
    'Highway': '#9e9e9e', 'Industrial Buildings': '#616161', 'Pasture': '#c5e1a5',
    'Permanent Crop': '#aed581', 'Residential Buildings': '#bcaaa4',
    'River': '#4fc3f7', 'SeaLake': '#0277bd'
}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def load_model(device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def classify_grid(image_path, model, device, patch_size):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    cols = width // patch_size
    rows = height // patch_size

    grid_labels = np.empty((rows, cols), dtype=object)

    patches = []
    coords = []
    for r in range(rows):
        for c in range(cols):
            left = c * patch_size
            top = r * patch_size
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patches.append(eval_transform(patch))
            coords.append((r, c))

    batch = torch.stack(patches).to(device)
    with torch.no_grad():
        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    for (r, c), pred in zip(coords, preds):
        grid_labels[r, c] = CLASS_NAMES[pred]

    return grid_labels, (rows, cols), img


def grid_to_rgb(grid_labels):
    rows, cols = grid_labels.shape
    rgb = np.zeros((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            hex_color = CLASS_COLORS[grid_labels[r, c]]
            rgb_val = tuple(int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5))
            rgb[r, c] = rgb_val
    return rgb


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(device)
    print("Model loaded.")

    print(f"\nClassifying {BEFORE_LABEL} image...")
    before_grid, shape1, before_img = classify_grid(BEFORE_IMG, model, device, PATCH_SIZE)

    print(f"Classifying {AFTER_LABEL} image...")
    after_grid, shape2, after_img = classify_grid(AFTER_IMG, model, device, PATCH_SIZE)

    if shape1 != shape2:
        rows = min(shape1[0], shape2[0])
        cols = min(shape1[1], shape2[1])
        before_grid = before_grid[:rows, :cols]
        after_grid = after_grid[:rows, :cols]
        print(f"Note: image sizes differed slightly, cropped both grids to {rows}x{cols}")

    rows, cols = before_grid.shape
    total_patches = rows * cols

    # ---------- Change detection ----------
    forest_before = (before_grid == 'Forest')
    forest_lost = forest_before & (after_grid != 'Forest')

    n_forest_before = forest_before.sum()
    n_forest_lost = forest_lost.sum()
    pct_forest_lost = (n_forest_lost / n_forest_before * 100) if n_forest_before > 0 else 0

    print("\n--- RESULTS ---")
    print(f"Grid size: {rows} x {cols} = {total_patches} patches")
    print(f"Patches classified as Forest in {BEFORE_LABEL}: {n_forest_before}")
    print(f"Of those, no longer Forest in {AFTER_LABEL}: {n_forest_lost} ({pct_forest_lost:.1f}%)")

    print(f"\nWhat those lost-forest patches became in {AFTER_LABEL}:")
    lost_coords = np.argwhere(forest_lost)
    became = {}
    for r, c in lost_coords:
        cls = after_grid[r, c]
        became[cls] = became.get(cls, 0) + 1
    for cls, count in sorted(became.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count} patches")

    # ---------- Visualization ----------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].imshow(before_img)
    axes[0, 0].set_title(f"{BEFORE_LABEL} - Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(after_img)
    axes[0, 1].set_title(f"{AFTER_LABEL} - Original")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(grid_to_rgb(before_grid))
    axes[1, 0].set_title(f"{BEFORE_LABEL} - Classified Land Cover")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(grid_to_rgb(after_grid))
    axes[1, 1].set_title(f"{AFTER_LABEL} - Classified Land Cover")
    axes[1, 1].axis('off')

    patches_legend = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_NAMES]
    fig.legend(handles=patches_legend, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig('land_cover_comparison.png', bbox_inches='tight', dpi=150)
    print("\nSaved land_cover_comparison.png")

    # ---------- Forest loss highlight map ----------
    fig2, ax = plt.subplots(figsize=(10, 8))
    highlight = np.zeros((rows, cols, 3))
    highlight[:, :] = [0.9, 0.9, 0.9]  # grey background
    highlight[forest_before] = [0.2, 0.6, 0.2]  # green = still relevant forest baseline
    highlight[forest_lost] = [0.9, 0.1, 0.1]  # red = forest lost

    ax.imshow(highlight)
    ax.set_title(f"Forest Loss Detected: {BEFORE_LABEL} \u2192 {AFTER_LABEL}\n"
                 f"Red = was Forest, no longer Forest ({n_forest_lost} of {n_forest_before} patches, {pct_forest_lost:.1f}%)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('forest_loss_map.png', dpi=150)
    print("Saved forest_loss_map.png")

    with open('deforestation_report.txt', 'w') as f:
        f.write(f"Deforestation Detection Report\n")
        f.write(f"Comparing {BEFORE_LABEL} vs {AFTER_LABEL}\n\n")
        f.write(f"Grid size: {rows} x {cols} = {total_patches} patches\n")
        f.write(f"Patches classified as Forest in {BEFORE_LABEL}: {n_forest_before}\n")
        f.write(f"No longer Forest in {AFTER_LABEL}: {n_forest_lost} ({pct_forest_lost:.1f}%)\n\n")
        f.write("What lost-forest patches became:\n")
        for cls, count in sorted(became.items(), key=lambda x: -x[1]):
            f.write(f"  {cls}: {count} patches\n")
    print("Saved deforestation_report.txt")
    print("\nDone. Send back: deforestation_report.txt, land_cover_comparison.png, forest_loss_map.png")


if __name__ == '__main__':
    main()