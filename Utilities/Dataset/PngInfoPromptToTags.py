from PIL import Image, PngImagePlugin
import csv
import json
import traceback
import argparse
import os
from glob import glob # Search for files
import pathlib # File pathing
import re # Regex

datasetPath = r"F:\StableDiffusion\Datasets\3 Pending\RJeezz\1_Style"

# For ease of use - reusing the tag file from the eva tagger. The Eva02 model
# (and its selected_tags.csv) now lives under TagManagement/Models/. Resolve the
# project root from this file (two levels up) so the path works no matter what
# directory the script is launched from.
CSV_FILE = "selected_tags.csv"
TAGGER_PATH = str(pathlib.Path(__file__).resolve().parents[2] / "TagManagement" / "Models") + "\\"
TAGGERS = [ "wd-eva02-large-tagger-v3" ]
CSV_HEADER_ERROR = "Unexpected .csv header format detected: "

def extract_metadata(image):
    if image is None:
        return "Invalid Image Path.", {}

    try:
        metadata = {}
        if 'metadata' in image.info:
            metadata = json.loads(image.info['metadata'])
        elif 'prompt' in image.info:
            metadata = json.loads(image.info['prompt'])
        elif 'Comment' in image.info:
            metadata = json.loads(image.info['Comment'])
            metadata['model'] = 'NovelAI'
        elif 'parameters' in image.info:
            if image.info['parameters'].startswith('{'):
                parameters_data = json.loads(image.info['parameters'])
                if 'sui_image_params' in parameters_data:
                    sui_image_params = parameters_data['sui_image_params']
                    metadata.update(sui_image_params)
                else:
                    metadata = parameters_data
            else:
                lines = image.info['parameters'].split(',')
                
                nPromptIdx = -1
                configIdx = -1

                for idx in range(len(lines)):
                    if 'Negative prompt: ' in lines[idx]:
                        nPromptIdx = idx
                    elif 'Steps: ' in lines[idx]:
                        configIdx = idx
                        if nPromptIdx == -1:
                            nPromptIdx = configIdx
                        break
                
                prompt = ','.join(lines[0:nPromptIdx]).strip()
                negative_prompt = '\n'.join(lines[nPromptIdx:configIdx]).replace('Negative prompt:', '').strip()

                #prompt = lines[0].strip()
                #negative_prompt = lines[1].strip().replace('Negative prompt:', '').strip()
                metadata['prompt'] = prompt
                metadata['negative_prompt'] = negative_prompt
                
                for line in lines[configIdx:]:
                    line = line.strip()
                    if line.startswith('Steps:'):
                        steps_info = line.split(':', 1)[1].strip().split(',')
                        for info in steps_info:
                            info = info.strip()
                            if ':' in info:
                                key, value = info.split(':', 1)
                                metadata[key.strip()] = value.strip()
        else:
            return "No supported metadata found in the image.", {}

        return "Metadata extracted successfully.", metadata
    except Exception as e:
        error_message = f"Error extracting metadata: {str(e)}\n{traceback.format_exc()}"
        return error_message, {}

def process_image(imagePath):
    image = Image.open(imagePath)
    status, metadata = extract_metadata(image)
    return status, metadata

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


def ProcessDataset(datasetPath, fileType = '.jpg'):
    print("Start Processing For: " + datasetPath)
    # Gather image files corresponding to the images in the specified folder
    fileTypePattern = '*' + fileType
    filePattern = (pathlib.Path(datasetPath) / fileTypePattern)
    datasetPaths = [*glob(str(filePattern))] # Get all filepaths fitting to file pattern

    logs = {}
    # For each text file
    for imgPath in datasetPaths:
        status, metadata = process_image(imgPath)
        filename = imgPath.split('\\')[-1]
        if metadata == {}:
            logs[filename] = status
        else:
            #print(metadata)
            generalTags, characterTags = LoadTagLists()
            # Remove Tags Not In Danbooru
            tags = []
            removed = []

            for tag in metadata['prompt'].split(','):
                # Converting to danbooru format for checks
                tag = tag.strip().replace(" ", "_").replace("\\", "")

                # Preprocess to remove emphasis
                # Remove surrounding parentheses if they wrap the whole word
                if tag.startswith('(') and tag.endswith(')'):
                    tag = tag[1:-1]
                    # Remove ':number' if it exists
                    tag = re.sub(r":\d+(\.\d+)?", "", tag)

                if tag in generalTags:
                    tags.append(tag)
                elif tag in characterTags:
                    tags.insert(0, tag)
                #else:
                #    removed.append(tag)
            # Write out a text file of the tags
            with open(os.path.splitext(imgPath)[0] + ".txt", "wt", encoding="utf-8") as f:
                f.write(", ".join(tags) + "\n")
                print("Processed: " + filename)
    print("Files with error loading metadata: ", len(logs.keys()))
    print(logs.keys())

ProcessDataset(datasetPath, 'png')