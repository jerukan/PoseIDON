from io import BytesIO
import os
import sys
from pathlib import Path
import requests
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Suppress warning messages
# warnings.filterwarnings("ignore")


def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)


def display_image_with_masks(
    image,
    masks,
    boxes,
    scores,
    mask_scores,
    figwidth=15,
    savefig=None,
    all_masks=True,
    show_confidence=True,
    show=True,
):
    if not all_masks:
        masks = masks[:1]
        boxes = boxes[:1]
        scores = scores[:1]
        mask_scores = mask_scores[:1]
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(figwidth, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image with Bounding Boxes")
    axes[0].axis("off")

    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(
            score.item(), 2
        )  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        axes[0].add_patch(rect)

        # Add confidence score as text
        if show_confidence:
            axes[0].text(
                x_min + 5,
                y_min + 5,
                f"Confidence: {confidence_score}",
                fontsize=8,
                color="red",
                verticalalignment="top",
            )

    for i, mask_np in enumerate(masks):
        axes[i + 1].imshow(mask_np, cmap="gray")
        axes[i + 1].set_title(f"Mask {i+1} (bbox score: {round(scores[i].item(), 2)}),\n(mask score: {round(mask_scores[i].item(), 2)})")
        axes[i + 1].axis("off")

    plt.tight_layout()
    if savefig is not None:
        fig.savefig(savefig)
    if show:
        plt.show()
    else:
        plt.close(fig)


def display_image_with_boxes(image, boxes, logits, show=True):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis("off")

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(
            logit.item(), 2
        )  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(
            x_min,
            y_min,
            f"Confidence: {confidence_score}",
            fontsize=8,
            color="red",
            verticalalignment="top",
        )
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")


def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")


def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")
