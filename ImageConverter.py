from glob import glob # Search for files
from tqdm import tqdm # Progress bar
import pathlib # File pathing
import cv2 # Image Resampling

# For AVIF
from PIL import Image
import pillow_avif

# Image folder path
datasetPath = r"F:\StableDiffusion\Datasets\3 Pending\Thomasz\Temp"
# Path to save to
savePath = r"F:\StableDiffusion\Datasets\3 Pending\Thomasz\Temp"

# File type to load
fileFormat = ".avif"
# File type to save as
saveType = '.png'

# Return a list of filepaths to each image in the specified folder
def GatherImagePaths(hrDatasetFolderName, fileType = '.png'):
    fileTypePattern = '*' + fileType
    filePattern = (pathlib.Path(hrDatasetFolderName) / fileTypePattern)
    datasetPaths = [*glob(str(filePattern))] # Get all filepaths fitting to file pattern
    return datasetPaths

# Create a dataset of a specific file type
def CreateDatasetOfType(datasetPaths, savePath, fileType = '.png'):
    # For each file in the dataset
    for imagePath in tqdm(datasetPaths): # Using tqdm to display a progress bar
        # Construct the save path for the current image
        fileName = pathlib.Path(imagePath).stem
        save = savePath + fileName + fileType
        if fileFormat != ".avif":
            img = OpenImageRGB(imagePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save, img)
        else:
            img = Image.open(imagePath)
            img.save(save)
        
def OpenImageRGB(imagePath):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Gather image paths
datasetPaths = GatherImagePaths(datasetPath, fileFormat)
print(datasetPath)
print(datasetPaths[0])

# Create a downscaled Low Resolution dataset using the High Resolution dataset
CreateDatasetOfType(datasetPaths, savePath, saveType)