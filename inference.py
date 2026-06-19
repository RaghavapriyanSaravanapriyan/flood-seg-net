import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SM_FRAMEWORK"] = "tf.keras"

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")

# ==========================================================
# Config
# ==========================================================
MODEL_PATH = "best_model_colab.keras"
MASK_DIR = "data/Mask"

IMG_SIZE = 256
THRESHOLD = 0.5

# ==========================================================
# Load model
# ==========================================================
print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "iou_score": sm.metrics.IOUScore(),
        "f1-score": sm.metrics.FScore()
    },
    compile=False
)

print("Model loaded successfully.\n")


# ==========================================================
# Prediction
# ==========================================================
def predict(image_path):

    original = cv2.imread(image_path)

    if original is None:
        raise FileNotFoundError(image_path)

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    h, w = original.shape[:2]

    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    inp = resized.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred = model.predict(inp, verbose=0)

    pred = pred[0, :, :, 0]

    pred = (pred > THRESHOLD).astype(np.uint8) * 255

    pred = cv2.resize(
        pred,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return original, pred


# ==========================================================
# Metrics
# ==========================================================
def compute_metrics(gt_mask, pred_mask):

    gt = (gt_mask > 127).astype(np.uint8)
    pred = (pred_mask > 127).astype(np.uint8)

    intersection = np.sum(gt * pred)

    dice = (2 * intersection + 1e-7) / (
        np.sum(gt) + np.sum(pred) + 1e-7
    )

    iou = (intersection + 1e-7) / (
        np.sum(gt) + np.sum(pred) - intersection + 1e-7
    )

    return dice, iou


# ==========================================================
# Error map
# ==========================================================
def make_error_map(gt_mask, pred_mask):

    gt = gt_mask > 127
    pred = pred_mask > 127

    error_map = np.zeros(
        (gt.shape[0], gt.shape[1], 3),
        dtype=np.uint8
    )

    # Green = TP
    error_map[np.logical_and(gt, pred)] = [0, 255, 0]

    # Red = FP
    error_map[np.logical_and(~gt, pred)] = [255, 0, 0]

    # Blue = FN
    error_map[np.logical_and(gt, ~pred)] = [0, 0, 255]

    return error_map


# ==========================================================
# Visualization
# ==========================================================
def save_results(image_path):

    original, pred_mask = predict(image_path)

    base = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    gt_path = os.path.join(
        MASK_DIR,
        base + ".png"
    )

    gt_mask = None

    if os.path.exists(gt_path):

        gt_mask = cv2.imread(
            gt_path,
            cv2.IMREAD_GRAYSCALE
        )

        gt_mask = cv2.resize(
            gt_mask,
            (
                original.shape[1],
                original.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # Overlay
    overlay = original.copy()

    flood_color = np.array([0, 100, 255])

    overlay[pred_mask > 0] = (
        overlay[pred_mask > 0] * 0.4
        + flood_color * 0.6
    ).astype(np.uint8)

    # Save mask
    mask_path = f"{base}_mask.png"
    cv2.imwrite(mask_path, pred_mask)

    # ======================================================
    # Plot
    # ======================================================
    if gt_mask is not None:

        error_map = make_error_map(
            gt_mask,
            pred_mask
        )

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(25, 5)
        )

        axes[0].imshow(original)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        axes[3].imshow(error_map)
        axes[3].set_title("Error Map")
        axes[3].axis("off")

        axes[4].imshow(overlay)
        axes[4].set_title("Overlay")
        axes[4].axis("off")

        dice, iou = compute_metrics(
            gt_mask,
            pred_mask
        )

    else:

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        axes[0].imshow(original)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

    plt.suptitle(
        "Flood Area Segmentation",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    viz_path = f"{base}_visualization.png"

    plt.savefig(
        viz_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    # ======================================================
    # Stats
    # ======================================================
    print("\n==============================")

    print("Mask saved:")
    print(mask_path)

    print("\nVisualization saved:")
    print(viz_path)

    print(
        f"\nFlood coverage: {(pred_mask > 0).mean()*100:.2f}%"
    )

    if gt_mask is not None:

        print(f"Dice score : {dice:.4f}")
        print(f"IoU score  : {iou:.4f}")

        print("\nError Map:")
        print("Green = True Positive")
        print("Red   = False Positive")
        print("Blue  = False Negative")

    print("==============================\n")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            "python inference.py data/Images/25.jpg\n"
        )

        sys.exit()

    image_path = sys.argv[1]

    if not os.path.exists(image_path):

        print("Image not found:")
        print(image_path)

        sys.exit()

    print("Running inference on:")
    print(image_path)

    save_results(image_path)