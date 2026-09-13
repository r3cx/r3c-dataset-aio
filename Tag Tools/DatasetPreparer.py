import argparse
import os

# Score tags for pony
SCORE = ""

# Trigger tag if required
TRIGGER = ""

# Old trigger tag to be removed
OLD_TRIGGER = ""

# Remove tags that add no value
NEGATIVES = "transparent background, unknown, official alternate costume, alternate costume, alternate hairstyle, alternate breast size, official alternate hairstyle, alternate hair length, alternate eye color, alternate hair color, virtual youtuber, score 9, score 8 up, score 7 up, score 6 up, score 5 up, score 4 up, parody, style parody, score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"

def RemoveDuplicateTags(tagList):
    # Resplit the tags before dupe check in case score / kw had delimited tags
    tagList = [tagList.strip() for tagList in ", ".join(tagList).split(",")]
    # Remove duplicates while maintaining the order
    unique_tags = []
    for tag in tagList:
        if tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags

def PrunePrecedence(tags, precedenceTags, exceptionTags):
    # If any of the tags of the image is part of the exception tags - skip
    if any(tag in exceptionTags for tag in tags):
        return tags
    # If any of the precedence tag is in the image tags, remove it
    for ptag in precedenceTags: # Check from the largest tag
        if ptag in tags:
            precedenceTags.remove(ptag) # Remove the largest tag and stop
            break
    # Return a list of tags containing only the largest occurence 
    return [tag for tag in tags if tag not in precedenceTags]

def PrunePreceedingTags(tags):
    # Male
    tags = PrunePrecedence(tags, ["gigantic testicles","huge testicles","large testicles","small testicles","testicles"], ["multiple penises", "multiple boys"])
    tags = PrunePrecedence(tags, ["gigantic penis","huge penis","large penis","small penis","penis"], ["multiple penises", "multiple boys"])
    
    # Female
    tags = PrunePrecedence(tags, ["gigantic breasts","huge breasts","large breasts","medium breasts","small breasts","flat chest","breasts"], ["multiple girls"])
    tags = PrunePrecedence(tags, ["huge nipples","small nipples","nipples"], ["multiple girls"])
    tags = PrunePrecedence(tags, ["large areolae","areolae"], ["multiple girls"])
    tags = PrunePrecedence(tags, ["huge ass","ass"], ["multiple girls"])
    tags = PrunePrecedence(tags, ["thick thighs","thighs"], ["multiple girls"])
    
    # Hair
    tags = PrunePrecedence(tags, ["absurdly long hair", "very long hair","long hair"], ["multiple girls"])
    tags = PrunePrecedence(tags, ["long bangs", "short bangs","bangs"], ["multiple girls"])

    return tags

def RemoveNegativeTags(tags, negativeTags):
    # Remove negative tags from the image tags
    return [tag for tag in tags if tag not in negativeTags]

# Potential issue if there is a need to escape multiple pairs of parentheses
def escapeParenthesisAndReplaceUnderscore(tags):
    modified_list = []
    for item in tags:
        # Add an \ before any parenthesis without \ before it
        item = item.replace('(', r'\(') if r'\(' not in item else item
        item = item.replace(')', r'\)') if r'\)' not in item else item

        # Replace underscores with spaces
        item = item.replace("_", " ")
        modified_list.append(item)

    return modified_list

def ProcessDataset(datasetPath, trigger, old_trigger, quality, undesired):
    print("Start Processing For: " + datasetPath)
    # Gather text files corresponding to the images in the specified folder
    txt_files = [f for f in os.listdir(datasetPath) if f.endswith(".txt")]
    # Record some stats of the current processing
    tagStats = []
    # For each text file
    for txt in txt_files:
        with open(os.path.join(datasetPath, txt), 'r', encoding='utf-8') as f:
            # Read the contents of the file and split by commas
            tags = [tag.strip() for tag in f.read().split(",")]

            # Escape Parentheses and replace underscores
            tags = escapeParenthesisAndReplaceUnderscore(tags)

            # Remove negative tags
            tags = RemoveNegativeTags(tags, undesired)

            # Prune preceeding tags
            tags = PrunePreceedingTags(tags)

            # Add score tags
            if quality != "":
                tags.insert(0, quality)

            # Remove old trigger tag if present
            if old_trigger != "":
                tags = [tag for tag in tags if tag != old_trigger]

            # Add trigger tags
            if trigger != "":
                tags.insert(0, trigger)

            # Remove duplicates while maintaining the order
            tags = RemoveDuplicateTags(tags)

        # Write the tags back to the file, separated by commas
        with open(os.path.join(datasetPath, txt), 'w', encoding='utf-8') as f:
            f.write(", ".join(tags))

        print("Tag Count: " + str(len(tags)) + " | Processed: " + txt)
        tagStats.append(len(tags))

    print("Updated: " + str(len(txt_files)) + " files")
    print("Average Tag Count: " + str(sum(tagStats) / len(tagStats)))
    print("Highest Tag Count: " + str(max(tagStats)))
    print("Lowest Tag Count: " + str(min(tagStats)))
    print()
    #input("Continue? ")
    print()

def setupArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory of images to be tagged")
    parser.add_argument("--trigger_tag", type=str, default=TRIGGER, help="Inserts this activation tag into the front of each tag file (Default: No activation tag)")
    parser.add_argument("--old_trigger_tag", type=str, default=OLD_TRIGGER, help="Removes this activation tag from the front of each tag file (Default: No old activation tag)")
    parser.add_argument("--quality_tags", type=str, default=SCORE, help="Comma-separated list of quality tags to add to the output (Default: PonyXL score tags)")
    parser.add_argument("--undesired_tags", type=str, default=NEGATIVES, help="Comma-separated list of tags to remove from the output (Default: Refer to DatasetPreparer.py for list of tags)")
    return parser

if __name__ == "__main__":
    parser = setupArgumentParser()
    args = parser.parse_args()

    ProcessDataset(args.data_dir, args.trigger_tag, args.old_trigger_tag, args.quality_tags, args.undesired_tags)