import os
import shutil

USE_FOLDER_KEY = True

def copy_and_rename_images(root_path, output_path):
    """
    Copies images from subfolders in the root_path into the specified output_path,
    renaming them with their folder name prefixed.
    
    Parameters:
        root_path (str): Path to the root folder containing subfolders with images.
        output_path (str): Path to the output folder where renamed images will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # if using folder key values instead of folder names
    key = 0
    
    # Iterate over all subfolders in the root folder
    folders = os.listdir(root_path)
    for folder_name in folders:
        folder_path = os.path.join(root_path, folder_name)
        
        # Check if it is a directory
        if os.path.isdir(folder_path):
            #print(f"Number of Files: " + str(len(os.listdir(folder_path))))
            # Increment key per folder
            key += 1
            counter = 0

            # Iterate over files in the subfolder
            for file_name in os.listdir(folder_path):
                # Check if the file is an image (e.g., png or jpg)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # Build source file path
                    source_file_path = os.path.join(folder_path, file_name)

                    # For Kemono DS
                    #folder_name = str(folder_name).split("_")[0]
                    
                    # Build destination file path with prefixed name
                    new_file_name = f"{folder_name}-{file_name}" 
                    # Overwrite the name
                    if USE_FOLDER_KEY:
                         root, extension = os.path.splitext(file_name)
                         new_file_name = f"{key}-{counter}{extension}" 
                         counter += 1
                
                    destination_file_path = os.path.join(output_path, new_file_name)
                    
                    # Copy the file to the output folder with the new name
                    shutil.copy2(source_file_path, destination_file_path)
                    print(f"Copied: {source_file_path} -> {destination_file_path}")
    
    print(f"All images have been copied and renamed to '{output_path}'.")

# usage
base_path = "F:\\StableDiffusion\\Datasets\\3 Pending\\fujihan_(osamuraifuji)"
root_path = base_path + "\\Source"
output_path = base_path + "\\To_Sort"
copy_and_rename_images(root_path, output_path)