import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Arquivo multi-label do TotalSegmentator
SEGMENTATION_PATH = "0801_clean_fdk_256.nii"

# Arquivo de dose gerado pela simulação
DOSE_PATH = "Dose-total.mhd"

OUTPUT_DIR = "resultados_dvh"
MASK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mascaras_binarias")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)


# ============================================================
# LABELS DOS ÓRGÃOS PRINCIPAIS - TOTALSEGMENTATOR
# ============================================================
# Labels oficiais usados:
#
# 10 = lung_upper_lobe_left
# 11 = lung_lower_lobe_left
# 12 = lung_upper_lobe_right
# 13 = lung_middle_lobe_right
# 14 = lung_lower_lobe_right
# 15 = esophagus
# 16 = trachea
# 51 = heart
# 79 = spinal_cord

ORGAN_LABELS = {
    "pulmao_esquerdo": [10, 11],
    "pulmao_direito": [12, 13, 14],
    "coracao": [51],
    "esofago": [15],
    "traqueia": [16],
    "medula_espinhal": [79],
}


# ============================================================
# CARREGAMENTO DOS ARQUIVOS
# ============================================================

def load_segmentation_nifti(path):
    """
    Carrega uma segmentação multi-label em formato NIfTI.

    Retorna:
    - segmentation: array NumPy com labels inteiros
    - affine: matriz affine original
    - header: cabeçalho original
    """
    nii = nib.load(path)

    segmentation = nii.get_fdata()
    segmentation = np.asarray(segmentation, dtype=np.int16)

    affine = nii.affine
    header = nii.header.copy()

    print("Segmentação carregada")
    print("Shape:", segmentation.shape)
    print("Tipo:", segmentation.dtype)

    return segmentation, affine, header


def load_dose_mhd(path):
    """
    Carrega um arquivo de dose em formato .mhd usando SimpleITK.

    Observação:
    SimpleITK normalmente retorna arrays no formato [z, y, x].
    NIfTI normalmente é trabalhado como [x, y, z].
    Por isso, pode ser necessário transpor os eixos.
    """
    image = sitk.ReadImage(path)

    dose = sitk.GetArrayFromImage(image)
    dose = np.asarray(dose, dtype=np.float32)

    print("\nDose carregada")
    print("Shape:", dose.shape)
    print("Tipo:", dose.dtype)
    print("Dose mínima:", np.min(dose))
    print("Dose máxima:", np.max(dose))
    print("Dose média global:", np.mean(dose))

    return dose


def show_available_labels(segmentation):
    """
    Mostra todos os labels presentes na segmentação,
    ignorando o fundo, normalmente label 0.
    """
    labels = np.unique(segmentation)
    labels = labels[labels != 0]

    print("\nLabels encontrados na segmentação:")
    print(labels)

    return labels


# ============================================================
# AJUSTE DE DIMENSÕES
# ============================================================

def match_shapes(segmentation, dose):
    """
    Compatibiliza as dimensões entre segmentação e dose.

    Casos comuns:
    - Segmentação NIfTI: [x, y, z]
    - Dose MHD via SimpleITK: [z, y, x]

    Se as dimensões forem compatíveis por transposição,
    a função ajusta automaticamente.
    """
    if segmentation.shape == dose.shape:
        print("\nShapes já compatíveis.")
        return segmentation, dose

    dose_transposed = dose.transpose(2, 1, 0)

    if segmentation.shape == dose_transposed.shape:
        print("\nAplicando transpose na dose: [z, y, x] -> [x, y, z]")
        dose = dose_transposed
        return segmentation, dose

    segmentation_transposed = segmentation.transpose(2, 1, 0)

    if segmentation_transposed.shape == dose.shape:
        print("\nAplicando transpose na segmentação: [x, y, z] -> [z, y, x]")
        segmentation = segmentation_transposed
        return segmentation, dose

    raise ValueError(
        "\nAs dimensões da segmentação e da dose não são compatíveis.\n"
        f"Shape da segmentação: {segmentation.shape}\n"
        f"Shape da dose: {dose.shape}\n\n"
        "Possíveis causas:\n"
        "1. A dose e a segmentação têm resoluções diferentes.\n"
        "2. A dose foi gerada com outro tamanho de voxel.\n"
        "3. A segmentação precisa ser reamostrada para a geometria da dose.\n"
        "4. Os arquivos não pertencem ao mesmo volume."
    )


# ============================================================
# MÁSCARAS
# ============================================================

def extract_binary_mask(segmentation, labels):
    """
    Extrai uma máscara binária a partir de um ou mais labels.

    Exemplo:
    pulmão esquerdo = labels [10, 11]
    pulmão direito = labels [12, 13, 14]
    """
    mask = np.isin(segmentation, labels)
    return mask


def save_binary_mask_nifti(mask, affine, header, organ_name, output_dir):
    """
    Salva uma máscara binária em formato NIfTI .nii.

    A máscara é salva como uint8:
    - 0 = fora do órgão
    - 1 = dentro do órgão
    """
    mask_uint8 = mask.astype(np.uint8)

    new_header = header.copy()
    new_header.set_data_dtype(np.uint8)

    mask_nii = nib.Nifti1Image(mask_uint8, affine, new_header)

    output_path = os.path.join(output_dir, f"mascara_{organ_name}.nii")

    nib.save(mask_nii, output_path)

    return output_path


# ============================================================
# DOSE POR ÓRGÃO
# ============================================================

def calculate_organ_dose(dose, mask):
    """
    Multiplica a máscara pela dose e extrai os valores de dose
    apenas dentro do órgão.
    """
    organ_dose = dose * mask

    values = dose[mask]

    # Remove NaN e infinitos, caso existam
    values = values[np.isfinite(values)]

    return organ_dose, values


def calculate_statistics(values):
    """
    Calcula estatísticas de dose dentro do órgão.
    """
    if values.size == 0:
        return {
            "voxels": 0,
            "dose_media": np.nan,
            "dose_min": np.nan,
            "dose_max": np.nan,
            "dose_mediana": np.nan,
            "dose_std": np.nan,
        }

    return {
        "voxels": int(values.size),
        "dose_media": float(np.mean(values)),
        "dose_min": float(np.min(values)),
        "dose_max": float(np.max(values)),
        "dose_mediana": float(np.median(values)),
        "dose_std": float(np.std(values)),
    }


# ============================================================
# DVH CUMULATIVO
# ============================================================

def generate_cumulative_dvh(values, bins=100):
    """
    Gera o DVH cumulativo.

    Para cada dose D, informa qual porcentagem do volume do órgão
    recebeu dose maior ou igual a D.
    """
    # Remove valores não finitos (NaN, inf) para não afetar os cálculos
    values = values[np.isfinite(values)]

    # Se não houver valores após a filtragem, retorna None para indicar
    # que não é possível gerar um DVH
    if values.size == 0:
        return None, None

    # Determina os limites mínimo e máximo de dose nos dados
    dose_min = float(np.min(values))
    dose_max = float(np.max(values))

    # Se todas as doses forem iguais, o DVH cumulativo é trivial:
    # uma única barra com 100% do volume recebendo essa dose
    if dose_min == dose_max:
        dose_bins = np.array([dose_min])
        cumulative_percent = np.array([100.0])
        return dose_bins, cumulative_percent

    # Calcula o histograma das doses usando o número de bins solicitado
    # 'hist' contém os contadores por bin e 'bin_edges' as bordas dos bins
    hist, bin_edges = np.histogram(
        values,
        bins=bins,
        range=(dose_min, dose_max)
    )

    # Constrói o DVH cumulativo: soma acumulada dos contadores a partir
    # do bin mais alto para o mais baixo. hist[::-1] inverte o array,
    # np.cumsum faz a soma acumulada e [::-1] reverte de volta à ordem
    cumulative = np.cumsum(hist[::-1])[::-1]

    # Converte contagens acumuladas em porcentagem do volume total
    cumulative_percent = cumulative / values.size * 100

    # Os pontos de dose associados aos contadores são as bordas iniciais
    # de cada bin (exclui a última borda que é o limite superior do último bin)
    dose_bins = bin_edges[:-1]

    # Retorna os valores de dose (bins) e a porcentagem cumulativa
    return dose_bins, cumulative_percent


def save_dvh_plot(dose_bins, cumulative_percent, organ_name, output_dir):
    """
    Salva o gráfico individual do DVH cumulativo em PNG.
    """

    plt.figure(figsize=(8, 6))

    # Volume em fração, de 0 a 1
    fractional_volume = cumulative_percent / 100.0

    plt.plot(dose_bins, fractional_volume, label=organ_name)
    plt.xlabel("Dose")
    plt.ylabel("Volume fracional")
    plt.title(f"DVH cumulativo - {organ_name}")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"dvh_{organ_name}.png")

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_dvh_csv(dose_bins, cumulative_percent, organ_name, output_dir):
    """
    Salva os dados do DVH cumulativo em CSV.
    """
    df = pd.DataFrame({
        "dose": dose_bins,
        "volume_percentual_acumulado": cumulative_percent,
        "volume_fracional_acumulado": cumulative_percent / 100.0
    })

    output_path = os.path.join(output_dir, f"dvh_{organ_name}.csv")

    df.to_csv(output_path, index=False)

    return output_path


def save_all_dvh_plot(dvh_data, output_dir, normalize_dose=False):
    """
    Salva um único gráfico com todos os DVHs cumulativos.

    Parâmetros:
    - dvh_data: lista de dicionários com:
        {
            "orgao": nome do órgão,
            "dose_bins": array de doses,
            "volume": array de volume acumulado em %
        }

    - normalize_dose:
        True  -> eixo X fica normalizado de 0 a 1, parecido com a imagem exemplo.
        False -> eixo X usa a dose real.
    """

    xlabel = "Dose"
    if len(dvh_data) == 0:
        print("Nenhum DVH disponível para gerar gráfico conjunto.")
        return None

    plt.figure(figsize=(10, 6))

    for item in dvh_data:
        organ_name = item["orgao"]
        dose_bins = item["dose_bins"]
        cumulative_percent = item["volume"]

        if dose_bins is None or cumulative_percent is None:
            continue

        if len(dose_bins) == 0:
            continue

        if normalize_dose:
            max_dose = np.max(dose_bins)

            if max_dose > 0:
                x = dose_bins / max_dose
            else:
                x = dose_bins

            xlabel = "Dose fracional (normalizada por dose máxima)"
        else:
            x = dose_bins
            xlabel = "Dose"

        y = cumulative_percent / 100.0

        plt.plot(x, y, linewidth=2, label=organ_name)

    plt.xlabel(xlabel)
    plt.ylabel("Volume fracional")
    plt.title("DVH cumulativo por órgão")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "dvh_todos_orgaos.png")

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_all_dvh_csv(dvh_data, output_dir, normalize_dose=True):
    """
    Salva todos os DVHs em um único CSV.

    Cada linha contém:
    - órgão
    - dose
    - dose_fracional
    - volume_percentual_acumulado
    - volume_fracional_acumulado
    """
    rows = []

    for item in dvh_data:
        organ_name = item["orgao"]
        dose_bins = item["dose_bins"]
        cumulative_percent = item["volume"]

        if dose_bins is None or cumulative_percent is None:
            continue

        if len(dose_bins) == 0:
            continue

        max_dose = np.max(dose_bins)

        for dose_value, volume_percent in zip(dose_bins, cumulative_percent):
            if normalize_dose and max_dose > 0:
                dose_fractional = dose_value / max_dose
            else:
                dose_fractional = dose_value

            rows.append({
                "orgao": organ_name,
                "dose": dose_value,
                "dose_fracional": dose_fractional,
                "volume_percentual_acumulado": volume_percent,
                "volume_fracional_acumulado": volume_percent / 100.0
            })

    if len(rows) == 0:
        return None

    df = pd.DataFrame(rows)

    output_path = os.path.join(output_dir, "dvh_todos_orgaos.csv")
    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    segmentation, affine, header = load_segmentation_nifti(SEGMENTATION_PATH)
    dose = load_dose_mhd(DOSE_PATH)

    show_available_labels(segmentation)

    segmentation, dose = match_shapes(segmentation, dose)

    results = []
    dvh_data = []

    for organ_name, labels in ORGAN_LABELS.items():
        print("\n" + "=" * 60)
        print(f"Processando órgão: {organ_name}")
        print(f"Labels usados: {labels}")

        mask = extract_binary_mask(segmentation, labels)

        voxel_count = int(np.sum(mask))

        if voxel_count == 0:
            print(
                f"Aviso: os labels {labels} não foram encontrados para {organ_name}.")
            continue

        # Salva máscara binária em .nii
        mask_path = save_binary_mask_nifti(
            mask=mask,
            affine=affine,
            header=header,
            organ_name=organ_name,
            output_dir=MASK_OUTPUT_DIR
        )

        # Calcula dose dentro do órgão
        organ_dose, values = calculate_organ_dose(dose, mask)

        # Estatísticas
        stats = calculate_statistics(values)

        stats["orgao"] = organ_name
        stats["labels"] = ",".join(map(str, labels))
        stats["arquivo_mascara"] = mask_path

        # DVH
        dose_bins, cumulative_percent = generate_cumulative_dvh(
            values,
            bins=100
        )

        if dose_bins is not None:
            # Guarda dados para o gráfico conjunto
            dvh_data.append({
                "orgao": organ_name,
                "dose_bins": dose_bins,
                "volume": cumulative_percent
            })

            # Salva gráfico individual
            plot_path = save_dvh_plot(
                dose_bins=dose_bins,
                cumulative_percent=cumulative_percent,
                organ_name=organ_name,
                output_dir=OUTPUT_DIR
            )

            # Salva CSV individual
            csv_path = save_dvh_csv(
                dose_bins=dose_bins,
                cumulative_percent=cumulative_percent,
                organ_name=organ_name,
                output_dir=OUTPUT_DIR
            )

            stats["arquivo_dvh_png"] = plot_path
            stats["arquivo_dvh_csv"] = csv_path
        else:
            stats["arquivo_dvh_png"] = None
            stats["arquivo_dvh_csv"] = None

        results.append(stats)

        print("Máscara salva em:", mask_path)
        print("Voxels do órgão:", stats["voxels"])
        print("Dose média:", stats["dose_media"])
        print("Dose mínima:", stats["dose_min"])
        print("Dose máxima:", stats["dose_max"])
        print("Dose mediana:", stats["dose_mediana"])
        print("Desvio padrão:", stats["dose_std"])

    if len(results) == 0:
        print("\nNenhum órgão principal foi encontrado na segmentação.")
        print("Verifique se o arquivo realmente é multi-label do TotalSegmentator.")
        return

    # ========================================================
    # SALVAR ESTATÍSTICAS GERAIS
    # ========================================================

    results_df = pd.DataFrame(results)

    columns_order = [
        "orgao",
        "labels",
        "voxels",
        "dose_media",
        "dose_min",
        "dose_max",
        "dose_mediana",
        "dose_std",
        "arquivo_mascara",
        "arquivo_dvh_png",
        "arquivo_dvh_csv",
    ]

    results_df = results_df[columns_order]

    output_stats_path = os.path.join(
        OUTPUT_DIR,
        "estatisticas_dose_por_orgao.csv"
    )

    results_df.to_csv(output_stats_path, index=False)

    # ========================================================
    # SALVAR DVH CONJUNTO
    # ========================================================

    all_dvh_plot_path = save_all_dvh_plot(
        dvh_data=dvh_data,
        output_dir=OUTPUT_DIR,
        normalize_dose=True
        # normalize_dose=True
    )

    all_dvh_csv_path = save_all_dvh_csv(
        dvh_data=dvh_data,
        output_dir=OUTPUT_DIR,
        normalize_dose=True
        # normalize_dose=True
    )

    print("\n" + "=" * 60)
    print("Processamento finalizado.")
    print("Estatísticas salvas em:", output_stats_path)
    print("Máscaras binárias salvas em:", MASK_OUTPUT_DIR)
    print("Arquivos DVH individuais salvos na pasta:", OUTPUT_DIR)
    print("DVH conjunto salvo em:", all_dvh_plot_path)
    print("CSV do DVH conjunto salvo em:", all_dvh_csv_path)


if __name__ == "__main__":
    main()
