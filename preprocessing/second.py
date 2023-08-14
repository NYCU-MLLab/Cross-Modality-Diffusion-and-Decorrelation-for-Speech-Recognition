import os
import shutil




# source_dir = r'C:\Users\chris\Music\temp'
original_path = r'\\?\E:\italy\temp'

# Set the path to your new directory
new_path = r'\\?\E:\italy\ccc'


# Create the new directory
os.makedirs(new_path, exist_ok=True)

# Loop through the directories in the original path
for dirpath, dirnames, filenames in os.walk(original_path):
    # Loop through the subdirectories
    for dirname in dirnames:
        # Create the new subdirectory in the new path
        new_subdir = os.path.join(new_path, dirname)
        os.makedirs(new_subdir, exist_ok=True)
        # Loop through the files in the subdirectory
        for filename in os.listdir(os.path.join(dirpath, dirname)):
            # Move the file to the new subdirectory
            old_filepath = os.path.join(dirpath, dirname, filename)
            new_filepath = os.path.join(new_subdir, dirname, filename)
            os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
            shutil.move(old_filepath, new_filepath)

# Remove the original directory
shutil.rmtree(original_path)

# input_folder_path = r"C:\Users\chris\Music\dirty"
# output_folder_path = r"C:\Users\chris\Music\temp"


        
# # Create the output folder if it doesn't exist
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

# # Loop through all WAV files in the input folder and its subdirectories
# for root, dirs, files in os.walk(input_folder_path):
#     for filename in files:
#         if filename.endswith(".wav"):
#             input_file_path = os.path.join(root, filename)
#             # Create the output subdirectory with the same name as the input file
#             output_subfolder_path = os.path.join(output_folder_path, os.path.splitext(filename)[0])
#             if not os.path.exists(output_subfolder_path):
#                 os.makedirs(output_subfolder_path, exist_ok=True)
#             # Move the input file to the output subdirectory
#             shutil.move(input_file_path, output_subfolder_path)

# # Copy the directory structure of the input folder to the output folder
# for root, dirs, files in os.walk(input_folder_path):
#     for dir in dirs:
#         output_dir_path = os.path.join(output_folder_path, os.path.relpath(os.path.join(root, dir), input_folder_path))
#         if not os.path.exists(output_dir_path):
#             os.makedirs(output_dir_path, exist_ok=True)
#     for file in files:
#         if not file.endswith(".wav"):
#             input_file_path = os.path.join(root, file)
#             output_file_path = os.path.join(output_folder_path, os.path.relpath(input_file_path, input_folder_path))
#             shutil.copy2(input_file_path, output_file_path)