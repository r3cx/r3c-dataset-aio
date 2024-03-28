# System Inputs
import csv
import argparse

# File I/O & Pathing
import os
import glob
from pathlib import Path

# Images
import cv2
from PIL import Image

# General
import numpy as np
from tqdm import tqdm

# For AI
import torch
from huggingface_hub import hf_hub_download
import onnxruntime

# Taggers Available For Use
# - SwinV2: a memory and GPU hog. Best metrics of the bunch
#   - wd-v1-4-swinv2-tagger-v2 / wd-swinv2-tagger-v3

# - ViT: fastest of the bunch, slightly less then stellar metrics
#   - wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2  / wd-vit-tagger-v3

# - ConvNext: nice performances, good metrics. A sweet spot
#   - wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2 / wd-convnext-tagger-v3

#   - Others:   wd-v1-4-moat-tagger-v2 / wd-v1-4-convnextv2-tagger-v2

'============================== CHANGE THIS PART ONLY ==============================' 
TAGGERS = [ # Set the taggers you're going to use in here
    #"wd-v1-4-swinv2-tagger-v2",
    #"wd-v1-4-convnextv2-tagger-v2",
    #"wd-v1-4-moat-tagger-v2",
    "wd-swinv2-tagger-v3"
    ]
'==================================================================================='

# from wd14 tagger
IMAGE_SIZE = 448

# Parsable image extensions
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

# Kaomojis to ignore during underscore parsing
DEFAULT_KAOMOJIS = '0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'

# HF repo files to be downloaded 
FILES = ["model.onnx", "selected_tags.csv"]

# For ease of use
CSV_FILE = FILES[-1]
DEFAULT_WD14_TAGGER_REPO = "wd-v1-4-swinv2-tagger-v2"
FILETYPE_TXT = ".txt"
TAGGER_PATH = "./Taggers/"

# String Constants
ERROR = "Error: "
FILE_OPEN_ERROR = "Failed to open file: "
CSV_HEADER_ERROR = "Unexpected .csv header format detected: "

# Preprocesses images by loading them into padded squares in BGR colorspace with resolution of IMAGE_SIZE
def PreprocessImage(image, imageSize):
    image = np.array(image)

    # Conver image from RGB -> BGR
    image = image[:, :, ::-1]  

    # Pad into a square
    size = max(image.shape[0:2])
    pad_x, pad_y = size - image.shape[1], size - image.shape[0]
    pad_l, pad_t = pad_x // 2, pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    # Fit the padded image to be within the size limits
    # Use INTER_AREA for shrinking, INTER_LANCZOS4 for expanding
    interp = cv2.INTER_AREA if size > imageSize else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (imageSize, imageSize), interpolation=interp)

    return image.astype(np.float32)

# Loads the required taggers, downloads them if unavailable
def LoadTaggers(enableForceDownload):
    models = []
    # Load each of the selected taggers - Download from SmilingWolf HF if not locally available
    for tagger in TAGGERS:
        # Download if required, else load from existing
        if not os.path.exists(f"{TAGGER_PATH}{tagger}") or enableForceDownload:
            print(f"Downloading WD14 tagger model from HF: SmilingWolf/{tagger}")
            # Download the specified files in the repo, inclusive of those in the subdirectories
            for file in FILES:
                hf_hub_download(f"SmilingWolf/{tagger}", file, cache_dir=f"{TAGGER_PATH}{tagger}", force_download=True, force_filename=file)
        else:
            print(f"Using existing WD14 tagger model: SmilingWolf/{tagger}")
        print(f"Loading tagger model: {tagger}")
       
        # Load onnx model
        modelPath = f"{TAGGER_PATH}{tagger}/{FILES[0]}"
        session = onnxruntime.InferenceSession(modelPath, None)
        models.append(session)
        
    # Returns the list of loaded models
    print(f"Completed loading of {len(TAGGERS)} tagger model(s)")
    return models

# Extracts the list of general and character tags from the .csv file that comes with the taggers
def LoadTagLists():
    # Only read labels from one tagger - They all use the same set of csv labels
    with open(os.path.join(f"{TAGGER_PATH}{TAGGERS[0]}", CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        # CSV Indexes: tag_id, name, category, count
        header, rows = l[0], l[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"{CSV_HEADER_ERROR}{header}"

    # Split up the tags (name) by the category
    generalTags = [row[1] for row in rows[1:] if row[2] == "0"]     # 0 - General
    characterTags = [row[1] for row in rows[1:] if row[2] == "4"]   # 4 - Character

    # Return our tag lists
    return generalTags, characterTags

def GatherImagePaths(dirPath, recursiveSearch):
    print(f"Accessing {dirPath}")
    imagePaths = []
    if recursiveSearch:
        for ext in IMAGE_EXTENSIONS:
            imagePaths += list(dirPath.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            imagePaths += list(dirPath.glob("*" + ext))
    imagePaths = list(set(imagePaths))
    imagePaths.sort()
    print(f"Found {len(imagePaths)} images.")
    return imagePaths

class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        # Load preprocesed image into a tensor
        try:
            image = Image.open(img_path).convert("RGB")
            image = PreprocessImage(image, IMAGE_SIZE)
            tensor = torch.tensor(image)
        except Exception as e:
            print(f"{FILE_OPEN_ERROR}{img_path}, {ERROR}{e}")
            return None
        return (tensor, img_path)

# To be used as the dataloader's collate function - when retrieving a batch of items, none tensors are discarded
def DiscardCorruptedDataInBatch(batch):
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

def SetupDataLoader(dataset, numWorkers):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=numWorkers,
        collate_fn=DiscardCorruptedDataInBatch,
        drop_last=False,
    )
    return dataloader

def RunInference(imageWPath, models, generalTags, characterTags, undesiredTags, tagFrequencies, generalThreshold, characterThreshold, debugPrint):
    # Obtain the images from the batch to be passed into the model for inference
    imageTensor = imageWPath[1]

    # inference using all the selected tagger models
    probabilityList = []
    for model in models:
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: [imageTensor]})[0]
        probabilityList.append(result)

    # Taking the average of all inference batches
    # TODO: Consider adding the capability to take a weighted average
    probability = np.mean(np.array(probabilityList), axis=0)[0]

    # First 4 list elements are for rating tags
    # - general, sensitive, questionable, explicit
    # Extract the probabilities of tags that come after these first 4
    tagProbabilities = probability[4:]

    # Compute the scores for each tagger on every tag
    imagePath = imageWPath[0]
    combined_tags = []
    general_tag_text = ""
    character_tag_text = ""
    # Iterate each of the probabilities in list, p, and the tag index i
    for i, p in enumerate(tagProbabilities): 
        # Check tag type by index and determine if probability passes threshold
        if i < len(generalTags) and p >= generalThreshold:
            tag_name = generalTags[i]
            if tag_name not in DEFAULT_KAOMOJIS:  # ignore emoji tags
                tag_name = tag_name.replace("_", " ")
            if tag_name not in undesiredTags:
                tagFrequencies[tag_name] = tagFrequencies.get(tag_name, 0) + 1
                general_tag_text += ", " + tag_name
                combined_tags.append(tag_name)
        elif i >= len(generalTags) and p >= characterThreshold:
            tag_name = characterTags[i - len(generalTags)]
            tag_name = tag_name.replace("_", " ")
            if tag_name not in undesiredTags:
                tagFrequencies[tag_name] = tagFrequencies.get(tag_name, 0) + 1
                character_tag_text += ", " + tag_name
                combined_tags.append(tag_name)

    # Remove leading comma from the taglist since we start insertions with ", "
    if len(general_tag_text) > 0:
        general_tag_text = general_tag_text[2:]
    if len(character_tag_text) > 0:
        character_tag_text = character_tag_text[2:]

    # Join the combined tags with ,
    tag_text = ", ".join(combined_tags)

    # Write the combined tags into a text file with the same name as the image
    with open(os.path.splitext(imagePath)[0] + FILETYPE_TXT, "wt", encoding="utf-8") as f:
        f.write(tag_text + "\n")
        if debugPrint:
            print(f"\n{imagePath}:\n  Character tags: {character_tag_text}\n  General tags: {general_tag_text}")

    # Return the tag frequencies for overall statistics record keeping
    return tagFrequencies

def StartAutoTagger(inputs):
    # Load/Download our required taggers
    models = LoadTaggers(inputs.force_download)

    # Load the list of general and character tags from the tagger csv
    generalTags, characterTags = LoadTagLists()

    # Gather image paths and load the images
    image_paths = GatherImagePaths(Path(inputs.data_dir), inputs.recursive_gather)

    undesiredTags = set(inputs.undesired_tags.split(","))

    # If the number of dataloaders is set, use a dataloader for faster loading
    if inputs.num_data_loader_workers is not None:
        dataPairs = SetupDataLoader(ImageLoadingPrepDataset(image_paths), inputs.num_data_loader_workers)
    else:
        dataPairs = [[(None, ip)] for ip in image_paths] # If no dataloader, map None to the path, load the file in runtime
    
    tagFrequencies = {}

    # Iterate the dataPairs and group them into batches before calling inference
    for data_entry in tqdm(dataPairs, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue
            # Split the current data into the image tensor and the file path
            image, image_path = data
            if image is not None: # If the image tensor is already loaded (From DataLoader)
                image = image.detach().numpy()
            else: # Otherwise try to load the image as a tensor
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = PreprocessImage(image)
                except Exception as e:
                    print(f"{FILE_OPEN_ERROR}{image_path}, {ERROR}{e}")
                    continue
            # Run inference on image
            tagFrequencies = RunInference((str(image_path), image), models, generalTags, characterTags, undesiredTags, tagFrequencies, inputs.general_threshold, inputs.character_threshold, inputs.debug_print)
    
    # If there is a need to print tag frequencies
    if inputs.frequency_tags:
        sorted_tags = sorted(tagFrequencies.items(), key=lambda x: x[1], reverse=True)
        print("\nTag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    print("Autotagging Completed!")
    print()
    #input("Enter anything to terminate")

def setupArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory of images to be tagged")
    parser.add_argument("--num_data_loader_workers", type=int, default=None, help="Number of workers to be used in the Torch DataLoader, set 0 to not use DataLoader")
    parser.add_argument("--threshold", type=float, default=0.35, help="Threshold of confidence to add a tag, tag confidence must be >= to threshold")
    parser.add_argument("--general_threshold", type=float, default=None, help="Threshold of confidence to add a tag for general category, same value as --threshold used if omitted")
    parser.add_argument("--character_threshold",type=float, default=None, help="Threshold of confidence to add a tag for character category, same value as --threshold used if omitted")
    parser.add_argument("--undesired_tags", type=str, default="", help="comma-separated list of undesired tags to remove from the output")
    parser.add_argument("--recursive_gather", action="store_true", help="If enabled, recursively gather images in subfolders of --data_dir")
    parser.add_argument("--frequency_tags", action="store_true", help="If enabled, print the frequency of tags across all tagged images")
    parser.add_argument("--force_download", action="store_true", help="If enabled, force download / redownload tagger models")
    parser.add_argument("--debug_print", action="store_true", help="If enabled, print the tag results for each image")
    return parser

# Enables use of dot.notation to access to dictionary attributes
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == "__main__":
    parser = setupArgumentParser()

    args = parser.parse_args()

    if args.general_threshold is None:
        args.general_threshold = args.threshold
    if args.character_threshold is None:
        args.character_threshold = args.threshold

    StartAutoTagger(DotDict(vars(args)))
