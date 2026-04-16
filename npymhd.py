import numpy as np
import SimpleITK as sitk
import os

def converter_npy_para_mhd(input_npy, output_mhd):
    # 1. Carregar matriz do arquivo NPY
    volume = np.load(input_npy)
    
    # 2. Ajustar valores: (CT * 1000 - 1000)
    volume = volume * 1000.0 - 1000.0
    
    # Converter para int16 (padrão em CT reais)
    volume = volume.astype(np.int16)
    
    # 3. Converter NumPy → SimpleITK
    image = sitk.GetImageFromArray(volume)
    
    # 4. Definir espaçamento arbitrário (1mm³)
    image.SetSpacing((1.0, 1.0, 1.0))  # ajuste se houver metadados reais
    
    # 5. Salvar como MHD/RAW
    sitk.WriteImage(image, output_mhd)
    print(f"Convertido: {input_npy} → {output_mhd}")

# --- Pasta de entrada e saída ---
input_folder = "tomografias_npy"
output_folder = "tomografias_mhd"
os.makedirs(output_folder, exist_ok=True)

# --- Converter os 10 arquivos ---
for i in range(0, 10):
    input_file = os.path.join(input_folder, f"000{i}_clean_fdk_256.npy")
    output_file = os.path.join(output_folder, f"000{i}_clean_fdk_256.mhd")
    converter_npy_para_mhd(input_file, output_file)
