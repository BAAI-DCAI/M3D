import os
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from collections import Counter
import unicodedata
import monai.transforms as mtf
from multiprocessing import Pool
from unidecode import unidecode

# input_dir = 'PATH/M3D_Cap/ct_quizze/'
# output_dir = 'PATH/M3D_Cap_npy/ct_quizze/'

input_dir = 'PATH/M3D_Cap/ct_case/'
output_dir = 'PATH/M3D_Cap_npy/ct_case/'

# Get all subfolders [00001, 00002....]
subfolders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]


transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])


def process_subfolder(subfolder):
    output_id_folder = os.path.join(output_dir, subfolder)
    input_id_folder = os.path.join(input_dir, subfolder)

    os.makedirs(output_id_folder, exist_ok=True)

    for subsubfolder in os.listdir(input_id_folder):
        if subsubfolder.endswith('.txt'):
            text_path = os.path.join(input_dir, subfolder, subsubfolder)
            with open(text_path, 'r') as file:
                text_content = file.read()

            search_text = "study_findings:"
            index = text_content.find(search_text)

            if index != -1:
                filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
            else:
                print("Specified string not found")
                filtered_text = text_content.replace("\n", " ").strip()


            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                search_text = "discussion:"
                index = text_content.find(search_text)
                if index != -1:
                    filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
                else:
                    print("Specified string not found")
                    filtered_text = text_content.replace("\n", " ").strip()


            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                filtered_text = text_content.replace("\n", " ").strip()


            new_text_path = os.path.join(output_dir, subfolder, subsubfolder)
            with open(new_text_path, 'w') as new_file:
                new_file.write(filtered_text)

        subsubfolder_path = os.path.join(input_dir, subfolder, subsubfolder)

        if os.path.isdir(subsubfolder_path):
            subsubfolder = unidecode(subsubfolder) # "Pöschl" -> Poschl
            output_path = os.path.join(output_dir, subfolder, f'{subsubfolder}.npy')

            image_files = [file for file in os.listdir(subsubfolder_path) if
                           file.endswith('.jpeg') or file.endswith('.png')]

            if len(image_files) == 0:
                continue

            image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            images_3d = []
            for image_file in image_files:
                image_path = os.path.join(subsubfolder_path, image_file)
                try:
                    img = Image.open(image_path)
                    img = img.convert("L")
                    img_array = np.array(img)
                    # normalization
                    img_array = img_array.astype(np.float32) / 255.0
                    images_3d.append(img_array[None])
                except:
                    print("This image is error: ", image_path)

            images_3d_pure = []
            try:
                img_shapes = [img.shape for img in images_3d]
                item_counts = Counter(img_shapes)
                most_common_shape = item_counts.most_common(1)[0][0]
                for img in images_3d:
                    if img.shape == most_common_shape:
                        images_3d_pure.append(img)
                final_3d_image = np.vstack(images_3d_pure)

                image = final_3d_image[np.newaxis, ...]

                image = image - image.min()
                image = image / np.clip(image.max(), a_min=1e-8, a_max=None)

                img_trans = transform(image)

                np.save(output_path, img_trans)
            except:
                print([img.shape for img in images_3d])
                print("This folder is vstack error: ", output_path)



with Pool(processes=32) as pool:
    with tqdm(total=len(subfolders), desc="Processing") as pbar:
        for _ in pool.imap_unordered(process_subfolder, subfolders):
            pbar.update(1)
