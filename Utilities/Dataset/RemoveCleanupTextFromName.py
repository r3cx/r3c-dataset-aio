import os # Search for files
from tqdm import tqdm # Progress bar
import pathlib # File pathing

# Image folder path
datasetPath = r"F:\StableDiffusion\Datasets\3 Pending\Suurin_(ksyaro)\1_Style"

# File type to load
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

# Return a list of filepaths to each image in the specified folder
def GatherImagePaths(dirPath, recursiveSearch):
    print(f"Accessing {dirPath}")
    imagePaths = []
    if recursiveSearch:
        for ext in IMAGE_EXTENSIONS:
            imagePaths += list(dirPath.glob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            imagePaths += list(dirPath.glob("*" + ext))
    imagePaths = list(set(imagePaths))
    imagePaths.sort()
    print(f"Found {len(imagePaths)} images.")
    return imagePaths

# Create a dataset of a specific file type
def RenameFilesContaining(datasetPaths, textToRemove, splitDelimiter):
    # For each file in the dataset
    for imagePath in tqdm(datasetPaths): # Using tqdm to display a progress bar
        imagePath = str(imagePath)
        if textToRemove in imagePath:
            splitPath = imagePath.split("\\")
            splitFile = splitPath[-1].split(".")
            splitFilename = splitFile[0].split(splitDelimiter)
            # Edit the filename
            splitFile[0] = splitDelimiter.join(splitFilename[0:-1])
            splitPath[-1] = ".".join(splitFile)
            path = "\\".join(splitPath)
            os.rename(imagePath, path)
        
        
# Gather image paths
datasetPaths = GatherImagePaths(pathlib.Path(datasetPath), True)
print(datasetPath)

# Rename the files
RenameFilesContaining(datasetPaths, "cleanup", "_")