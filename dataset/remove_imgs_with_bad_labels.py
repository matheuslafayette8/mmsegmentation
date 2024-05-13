import os
import shutil

base_path = "dataset/new_sub_datasets/"
datasets_name = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

for dataset_name in datasets_name:
    dataset_path = os.path.join(base_path, dataset_name)
    visualize_masks_path = os.path.join(dataset_path, "visualize_masks")
    rbg_images_path = os.path.join(dataset_path, "rgb_imgs")
    images_with_corrected_labels_path = os.path.join(dataset_path, "images_with_corrected_labels")
    
    # Criar a pasta 'images_with_corrected_labels' se ela não existir
    if not os.path.exists(images_with_corrected_labels_path):
        os.makedirs(images_with_corrected_labels_path)
    
    # Obter lista de arquivos em visualize_masks
    files_to_move = os.listdir(visualize_masks_path)
    
    # Mover os arquivos para images_with_corrected_labels com novo caminho
    for file_name in files_to_move:
        old_path = os.path.join(rbg_images_path, file_name)
        new_path = os.path.join(images_with_corrected_labels_path, file_name)
        shutil.copy(old_path, new_path)

print("Concluído! Imagens copiadas com labels corrigidas.")
