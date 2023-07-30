
import shutil
import os
from skew_correction.helper import get_images_in_dir, read_raw_image
from skew_correction.constants import angle2label
import random
import pandas as pd
import sys
from tqdm import tqdm


data_dir = "../data/"
org_data = "../data/original/"
rotated_data = "../data/rotated/"

def prepare_data(folder, multiple=1):
    """rotates image acc to specified logic and saves it. src folder must have images at 0 degrees."""
    label_counts = {k:0 for k in angle2label.keys()}
    src_dir = os.path.join(org_data, folder)
    trg_dir = os.path.join(rotated_data, folder)
    csv_data = []  
    extensions = ['.jpg', '.jpeg', '.png']
    src_filepaths = get_images_in_dir(src_dir, return_path=True)
    if len(src_filepaths) == 0:
        print(f"No images found in {src_dir}")
        return None
    
    for num in range(multiple):
        for file in tqdm(src_filepaths):
            img = read_raw_image(file)
            # print(f"src: {file}")
            angle = random.choice([0, 90, 180, 270])
            img = img.rotate(angle, expand=True)
            save_filename = f"{os.path.basename(file).split('.')[0]}-{angle}.jpg"
            trg_filepath = os.path.join(trg_dir, save_filename)
            # print(f"target: {trg_filepath}")
            img.save(trg_filepath)

            label_counts[angle] += 1
            csv_data.append((trg_filepath, angle))
    
    
    csv_data = pd.DataFrame(csv_data, columns=["path", "label"])
    csv_data.to_csv(os.path.join(data_dir,f"{folder}_data.csv"), index=False)
    print(f"A total of {sum(label_counts.values())} images were generated")
    print(f"Class wise counts of data generated for {folder} are: {label_counts}")
    return None

if __name__ == "__main__":
    folder = sys.argv[1]
    try:
        mulitple = int(sys.argv[2])
    except:
        mulitple = 1
    prepare_data(folder, mulitple)