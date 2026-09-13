from glob import glob
from tqdm import tqdm
import pathlib
import cv2
import random
import os

# Image folder path
datasetPath = r"F:\StableDiffusion\Datasets\2 Baking\Milimon\1_Style"
savePath = datasetPath + "\\flipped\\"
os.makedirs(savePath, exist_ok=True)

# Supported image types
supported_formats = ('.png', '.jpg', '.jpeg', '.webp')

# Return a list of filepaths for supported image types
def GatherImagePaths(hrDatasetFolderName):
    datasetPaths = []
    for ext in supported_formats:
        filePattern = str(pathlib.Path(hrDatasetFolderName) / f'*{ext}')
        datasetPaths.extend(glob(filePattern))
    return datasetPaths

# Open an image using OpenCV and convert to RGB
def OpenImageRGB(imagePath):
    img = cv2.imread(imagePath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply random flips and save
def RandomFlipDataset(datasetPaths, savePath):
    for imagePath in tqdm(datasetPaths):
        img = OpenImageRGB(imagePath)

        # Apply random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # Apply random vertical flip
        #if random.random() < 0.5:
        #    img = cv2.flip(img, 0)

        fileName = pathlib.Path(imagePath).name
        saveFullPath = os.path.join(savePath, fileName)

        # Save in the same format
        ext = pathlib.Path(fileName).suffix
        if ext.lower() in supported_formats:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(saveFullPath, img_bgr)

# Main
datasetPaths = GatherImagePaths(datasetPath)
print(f"Loaded {len(datasetPaths)} images from {datasetPath}")
print(f"Example: {datasetPaths[0]}")

RandomFlipDataset(datasetPaths, savePath)