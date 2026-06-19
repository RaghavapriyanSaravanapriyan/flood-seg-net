import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import sys 
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import matplotlib.pyplot as plt

# config
MODEL_PATH = "best_model_colab.keras"
IMG_SIZE   = 256  # internal processing size
THRESHOLD  = 0.5

# load best weights model
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'iou_score': sm.metrics.IOUScore(),
        'f1-score' : sm.metrics.FScore()
    },
    compile = False
)

# running inference
def predict(image_path):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original.shape[:2]

    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    inp = resized.astype(np.float32) / 255.0  # same as training
    inp = np.expand_dims(inp, axis=0)

    pred = model.predict(inp, verbose=0)
    pred = pred[0, :, :, 0]
    pred = (pred > THRESHOLD).astype(np.uint8) * 255

    mask_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return original, mask_full

# save result
def save_results(image_path):
    original, mask = predict(image_path)

    # save raw mask
    base      = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = f"{base}_mask.png"
    viz_path  = f"{base}_visualization.png"

    cv2.imwrite(mask_path, mask)

    # save side by side visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask", fontsize=14)
    axes[1].axis("off")

    # overlay — flood area highlighted in blue
    overlay          = original.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 100, 255]) * 0.6).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    plt.suptitle("Flood Area Segmentation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Mask saved      : {mask_path}")
    print(f"Visualization   : {viz_path}")
    print(f"Original size   : {original.shape[1]}x{original.shape[0]}")
    print(f"Flood pixels    : {(mask > 0).sum()}")
    print(f"Flood coverage  : {(mask > 0).mean() * 100:.1f}%")

# cli
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py data/Image/9.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File not found — {image_path}")
        sys.exit(1)

    print(f"Running inference on: {image_path}")
    save_results(image_path)