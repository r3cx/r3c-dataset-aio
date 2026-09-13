from glob import glob # Search for files
from tqdm import tqdm # Progress bar
import pathlib # File pathing
import cv2 # Image Resampling
import os

# For AVIF
from PIL import Image
import pillow_avif

# Image folder path
datasetPath = r"F:\StableDiffusion\Datasets\3 Pending\Mizumizuni\1_Style"
# Path to save to
savePath = "F:\StableDiffusion\Datasets\\1 Grabber_Outputs\Temp_Crop"

# File type to load
fileFormat = ".png"
# File type to save as
saveType = '.jpg' # not used

# Bottom X percent to crop off
cropPercent = 1.0

# Return a list of filepaths to each image in the specified folder
def GatherImagePaths(hrDatasetFolderName, fileType = '.png'):
    fileTypePattern = '*' + fileType
    filePattern = (pathlib.Path(hrDatasetFolderName) / fileTypePattern)
    datasetPaths = [*glob(str(filePattern))] # Get all filepaths fitting to file pattern
    return datasetPaths

# Create a dataset of a specific file type
def CreateCropDatasetOfType(datasetPaths, savePath, fileType = '.png'):
    # For each file in the dataset
    for imagePath in tqdm(datasetPaths): # Using tqdm to display a progress bar
        crop_bottom_x_percent(imagePath, savePath, cropPercent)

def crop_bottom_x_percent(image_path: str, output_dir: str, percent: float):
    """
    Load an image, crop out the bottom x% of the image, and save it to the output directory.
    
    Parameters:
        image_path (str): Path to the input image file.
        output_dir (str): Path to the output directory.
        percent (float): Percentage of the bottom part to crop (0-100).
    
    Returns:
        str: Path to the saved cropped image.
    """
    # Ensure the percent value is within a valid range
    if not (0 <= percent <= 100):
        raise ValueError("Percent must be between 0 and 100.")
    
    # Load the image
    img = Image.open(image_path)
    # Get image dimensions
    width, height = img.size
    # Calculate the new height after cropping
    new_height = int(height * (1 - percent / 100))
    # Crop the image (left, upper, right, lower)
    cropped_img = img.crop((0, 0, width, new_height))
    # Get the filename from the input path
    filename = os.path.basename(image_path)
    # Create the output path
    output_path = os.path.join(output_dir, filename)
    # Save the cropped image to the output directory
    cropped_img.save(output_path)
    return output_path

# Gather image paths
datasetPaths = GatherImagePaths(datasetPath, fileFormat)
print(len(datasetPath))

# Create a downscaled Low Resolution dataset using the High Resolution dataset
CreateCropDatasetOfType(datasetPaths, savePath, saveType)