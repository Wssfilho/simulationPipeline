"""
Conversor de imagens CT: .npy → .nii.gz
Uso: python npy_to_nii.py <arquivo.npy> [opções]

Exemplos:
  python npy_to_nii.py ct.npy
  python npy_to_nii.py ct.npy --spacing 1.0 1.0 2.5
  python npy_to_nii.py ct.npy --output minha_saida.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def calibrate_to_hu(arr, raw_ref1, hu_ref1, raw_ref2, hu_ref2, out_dtype=np.float32):
    """
    Calibração linear:
        HU = a * I + b
    onde I é a intensidade original.
    """

    if raw_ref2 == raw_ref1:
        raise ValueError("As duas referências brutas não podem ser iguais.")

    a = (hu_ref2 - hu_ref1) / (raw_ref2 - raw_ref1)
    b = hu_ref1 - a * raw_ref1

    print(f"Calibração linear: HU = {a:.6f} * I + {b:.6f}")

    arr_hu = a * arr + b
    return arr_hu.astype(out_dtype)


def npy_to_nii(
    input_path: str,
    output_path: str = None,
    spacing: tuple = (1.0, 1.0, 1.0),
    do_calibration=True,
    raw_air=0.0,
    hu_air=-1000.0,
    raw_soft=1.05,
    hu_soft=40.0
):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    if not input_path.suffix == ".npy":
        raise ValueError(
            f"Arquivo deve ter extensão .npy, recebido: {input_path.suffix}")

    # Define o caminho de saída (padrão: pasta 'resultados')
    if output_path is None:
        output_path = input_path.parent / \
            "imagens_ct" / f"{input_path.stem}.nii.gz"
    output_path = Path(output_path)

    print(f"[INFO] Carregando: {input_path}")
    data = np.load(str(input_path))
    data = data.transpose(2, 1, 0)
    data = data[:, ::-1, :]
    print(f"[INFO] Shape original: {data.shape}")
    print(f"[INFO] Dtype: {data.dtype}")
    print(f"[INFO] Valor mín/máx: {data.min():.1f} / {data.max():.1f}")

    # Garante array 3D (remove dimensões extras se necessário)
    if data.ndim == 4:
        print(
            f"[AVISO] Array 4D detectado, usando canal 0 (shape={data.shape})")
        data = data[0]
    elif data.ndim == 2:
        print(f"[AVISO] Array 2D detectado, adicionando dimensão Z")
        data = data[:, :, np.newaxis]
    elif data.ndim != 3:
        raise ValueError(
            f"Shape inválido: {data.shape}. Esperado 2D, 3D ou 4D.")

    if do_calibration:
        data = data.astype(np.float32)
        data = calibrate_to_hu(
            data,
            raw_ref1=raw_air,
            hu_ref1=hu_air,
            raw_ref2=raw_soft,
            hu_ref2=hu_soft,
            out_dtype=np.float32
        )

        print("Min / Max calibrados:", float(np.min(data)), float(np.max(data)))

    # Converte para int16 (padrão CT Hounsfield)
    if data.dtype != np.int16:
        print(f"[INFO] Convertendo dtype {data.dtype} → int16")
        data = data.astype(np.int16)

    # Monta a matrix affine com o espaçamento real dos voxels
    sx, sy, sz = spacing
    affine = np.diag([sx, sy, sz, 1.0])

    # Cria e salva imagem NIfTI
    nii_img = nib.Nifti1Image(data, affine)
    nii_img.header.set_zooms(spacing)
    nii_img.header.set_xyzt_units("mm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii_img, str(output_path))

    print(f"[OK] Salvo em: {output_path}")
    print(f"[OK] Shape final: {data.shape} | Spacing: {spacing} mm")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Converte imagens CT de .npy para .nii.gz (pronto para TotalSegmentator)"
    )
    parser.add_argument(
        "input",
        help="Caminho do arquivo .npy de entrada"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Caminho do arquivo .nii.gz de saída (padrão: pasta resultados/)"
    )
    parser.add_argument(
        "--spacing", "-s",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
        help="Espaçamento dos voxels em mm nos eixos X Y Z (padrão: 1.0 1.0 1.0)"
    )
    parser.add_argument(
        "--batch", "-b",
        default=None,
        metavar="PASTA",
        help="Converte todos os .npy de uma pasta de uma vez"
    )

    args = parser.parse_args()
    spacing = tuple(args.spacing)

    # Modo batch: converte todos os .npy de uma pasta
    if args.batch:
        pasta = Path(args.batch)
        arquivos = list(pasta.glob("*.npy"))
        if not arquivos:
            print(f"[AVISO] Nenhum arquivo .npy encontrado em: {pasta}")
            return
        print(f"[INFO] {len(arquivos)} arquivo(s) encontrado(s) em {pasta}")
        for arq in arquivos:
            try:
                npy_to_nii(str(arq), spacing=spacing)
            except Exception as e:
                print(f"[ERRO] {arq.name}: {e}")
        return

    # Modo arquivo único
    npy_to_nii(args.input, args.output, spacing)


if __name__ == "__main__":
    main()
