import os
import cv2

from inference import predict, compute_metrics, save_results

IMAGE_DIR = "data/Image"
MASK_DIR = "data/Mask"

MIN_FLOOD_COVERAGE = 0.10

results = []

print("Evaluating images...\n")

for image_name in os.listdir(IMAGE_DIR):

    image_path = os.path.join(IMAGE_DIR, image_name)

    base = os.path.splitext(image_name)[0]

    gt_path = os.path.join(MASK_DIR, base + ".png")

    if not os.path.exists(gt_path):
        continue

    # Predict
    _, pred_mask = predict(image_path)

    # Load ground truth
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    gt_mask = cv2.resize(
        gt_mask,
        (pred_mask.shape[1], pred_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Ignore nearly empty masks
    flood_coverage = (gt_mask > 127).mean()

    if flood_coverage < MIN_FLOOD_COVERAGE:
        continue

    dice, iou = compute_metrics(gt_mask, pred_mask)

    results.append({
        "image": image_name,
        "dice": dice,
        "iou": iou,
        "coverage": flood_coverage
    })

# Sort descending by Dice
results.sort(
    key=lambda x: x["dice"],
    reverse=True
)

print("=" * 70)
print("TOP 20 IMAGES")
print("=" * 70)

for i, r in enumerate(results[:20], start=1):

    print(
        f"{i:2d}. "
        f"{r['image']:15} "
        f"Dice={r['dice']:.4f} "
        f"IoU={r['iou']:.4f} "
        f"Coverage={r['coverage']*100:.1f}%"
    )

# Best image
best = results[0]

print("\n" + "=" * 70)
print("BEST IMAGE")
print("=" * 70)

print(f"Image      : {best['image']}")
print(f"Dice Score : {best['dice']:.4f}")
print(f"IoU Score  : {best['iou']:.4f}")
print(f"Coverage   : {best['coverage']*100:.1f}%")

best_image_path = os.path.join(
    IMAGE_DIR,
    best["image"]
)

print("\nGenerating visualization...")

save_results(best_image_path)