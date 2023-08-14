import os
import shutil


input_folder_path = r"\\?\E:\italy\clips"
# output_folder_path = r"\\?\D:\common_voice\temp"
output_folder_path = r"\\?\E:\italy\temp"
        
# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Loop through all WAV files in the input folder and its subdirectories
for root, dirs, files in os.walk(input_folder_path):
    for filename in files:
        if filename.endswith(".mp3"):
            input_file_path = os.path.join(root, filename)
            # Create the output subdirectory with the same name as the input file
            output_subfolder_path = os.path.join(output_folder_path, os.path.splitext(filename)[0])
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path, exist_ok=True)
            # Move the input file to the output subdirectory
            shutil.move(input_file_path, output_subfolder_path)

# Copy the directory structure of the input folder to the output folder
for root, dirs, files in os.walk(input_folder_path):
    for dir in dirs:
        output_dir_path = os.path.join(output_folder_path, os.path.relpath(os.path.join(root, dir), input_folder_path))
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
    for file in files:
        if not file.endswith(".mp3"):
            input_file_path = os.path.join(root, file)
            output_file_path = os.path.join(output_folder_path, os.path.relpath(input_file_path, input_folder_path))
            shutil.copy2(input_file_path, output_file_path)
            

# 000f7ff65feebae7288345f9230fe0684d99712d846dd6c7c01d03a138066267c4213a43727a6fe35df14e58e9bccfaac6747db547fbec4e1c53127048427090