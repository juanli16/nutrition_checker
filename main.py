from pathlib import Path
import torch
from torchvision import datasets
from model.helpers import plot
import matplotlib.pyplot as plt

ROOT = Path("data/nutritionverse-real/nutritionverse-manual/nutritionverse-manual")
IMAGE_PATH = str(ROOT / "images")
ANNOTATION_PATH = str(ROOT / "images" / "_annotations.coco.json")

def main():
    ds = datasets.CocoDetection(IMAGE_PATH, ANNOTATION_PATH)

    img, target = ds[0]
    print(target[0].keys())

    dataset = datasets.wrap_dataset_for_transforms_v2(ds, target_keys=("boxes", "labels", "masks"))

    print(dataset[0])
    plot([dataset[0], dataset[1]])
    plt.show()



if __name__ == "__main__":
    main()
