import os
import cv2
import numpy as np

background_value = 0
continuous_lane_value = 1
dashed_lane_value = 2
road_edge_value = 3

base_path = "dataset/sub_datasets/"
datasets_name = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
datasets_name = ["westgate"]
base_path = "dataset/new_sub_datasets/"

for dataset_name in datasets_name:
    base_path_dataset = f"dataset/new_sub_datasets/{dataset_name}/"

    input_path = base_path_dataset + "sem_seg_masks/"
    output_path = base_path_dataset + "visualize_masks/"
    low_i_path = base_path_dataset + "low_i_imgs/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    image_files = os.listdir(input_path)
    
    for image_file in image_files:
        image_path = os.path.join(input_path, image_file)
        low_i_img_path = os.path.join(low_i_path, image_file)
        
        # Carregar a imagem de fundo
        background_img = cv2.imread(low_i_img_path)
        
        # Ler a máscara em escala de cinza
        mask_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Criar uma máscara colorida com transparência
        mask_colored = np.zeros((mask_gray.shape[0], mask_gray.shape[1], 4), dtype=np.uint8)
        mask_colored[:, :, 3] = 255  # Definir o canal alpha para opacidade total
        
        # Mapear as cores para cada valor de intensidade na máscara
        mask_colored[mask_gray == continuous_lane_value] = [0, 0, 255, 75]  # Vermelho transparente para continuous_lane
        mask_colored[mask_gray == dashed_lane_value] = [0, 255, 0, 75]  # Verde transparente para dashed_lane
        mask_colored[mask_gray == road_edge_value] = [255, 0, 0, 75]  # Azul transparente para road_edge
        
        # Mesclar a máscara com a imagem de fundo usando a transparência
        result_image = cv2.addWeighted(background_img, 1, mask_colored[:, :, :3], 0.5, 0)
        
        # Salvar a imagem resultante
        output_image_path = os.path.join(output_path, image_file)
        cv2.imwrite(output_image_path, result_image)

print("Processo concluído. As imagens coloridas com transparência foram salvas em", base_path)