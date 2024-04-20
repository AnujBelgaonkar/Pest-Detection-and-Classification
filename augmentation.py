import Augmentor
import os
from PIL import Image
path_to_data = r"E:\Projects\Pest Detection and Classification using CNN\archive(1)"
target_path = r"E:\Projects\Pest Detection and Classification using CNN\Augmented"

import os
from PIL import Image

def convert_images_to_rgb_in_folders(root_dir):
    # Iterate through each folder in the root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        
        # Check if the item is a directory
        if os.path.isdir(folder_path):
            print(f"Converting images in folder: {folder_name}")
            
            # Iterate through each file in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # Check if the file is an image
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    try:
                        # Open the image
                        image = Image.open(file_path)
                        
                        # Convert to RGB if not already in RGB format
                        if image.mode != 'RGB':
                            rgb_image = image.convert('RGB')
                            
                            # Replace the original image with the RGB version
                            rgb_image.save(file_path)
                            print(f"Converted {file_name} to RGB and replaced the original.")
                        else:
                            print(f"{file_name} is already in RGB format. Skipping...")
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
    
    print("Conversion complete.")

# Example usage:
convert_images_to_rgb_in_folders(path_to_data)


folders = [ f.path for f in os.scandir(path_to_data) if f.is_dir() ]

target_folders = [f.path for f in os.scandir(target_path) if f.is_dir()]

combined = list(zip(folders,target_folders))

pipelines = []
for x in combined:
    print(f"Folder is: {x[0]}")
    pipelines.append(Augmentor.Pipeline(x[0],output_directory=x[1]))
    print("\n----------------------------\n")


for p in pipelines:
    p.rotate(probability=0.5, max_left_rotation=7, max_right_rotation=7)
    p.zoom(probability=0.5, min_factor=0.5, max_factor=0.9)
    p.shear(probability=0.6, max_shear_left = 0.7, max_shear_right = 0.7)
    p.flip_left_right(probability=0.4)
    p.flip_top_bottom(probability=0.8)
    p.rotate90(probability=0.1)
    p.flip_random(probability = 0.2)
    p.sample(400)
    p.process()




