#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converte volume .npy para NIfTI (.nii.gz).

Mantém a mesma lógica do npy2mhd.py:
  - calibração linear para Unidades Hounsfield (HU = slope * I + intercept)
  - ajuste da ordem dos eixos (zyx ou xyz)
  - máscara cilíndrica no plano x-y
  - limpeza de valores não físicos (min/max HU)
  - centralização via matriz affine
  - conversão de tipo (int16, float32, uint8)

Exemplos:

python npy2nii.py volume.npy --output volume.nii.gz

python npy2nii.py volume.npy --output volume.nii.gz \
    --spacing 1 1 1 \
    --origin -128 -128 -128 \
    --dtype int16

python npy2nii.py volume.npy --output volume.nii.gz \
    --array-order zyx \
    --mask-cylinder \
    --air-hu -1000 \
    --min-hu -1200

python npy2nii.py volume.npy --output volume.nii.gz \
    --spacing 0.977 0.977 2.5 \
    --hu-slope 990.48 \
    --hu-intercept -1000 \
    --mask-cylinder \
    --min-hu -1200 \
    --max-hu 3000 \
    --center
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib


# ──────────────────────────────────────────────────────────────────────────────
# Máscara cilíndrica
# ──────────────────────────────────────────────────────────────────────────────

def apply_cylindrical_air_mask(arr, spacing_xyz, air_hu=-1000.0, radius_mm=None):
    """
    Aplica uma máscara cilíndrica no plano x-y, zerando (com HU do ar)
    todos os voxels fora do cilindro.

    Parâmetros
    ----------
    arr : np.ndarray, shape (z, y, x)
        Volume em ordem numpy (z, y, x).
    spacing_xyz : tuple/list de 3 floats
        Espaçamento físico em mm na ordem (sx, sy, sz).
    air_hu : float
        Valor de HU atribuído à região externa ao cilindro.
    radius_mm : float ou None
        Raio do cilindro em mm. Se None, usa min(Nx*sx, Ny*sy) / 2.

    Retorna
    -------
    arr : np.ndarray
        Volume com a máscara aplicada (in-place).
    radius_mm : float
        Raio efetivamente utilizado.
    """
    nz, ny, nx = arr.shape
    sx, sy, _ = spacing_xyz

    # Coordenadas físicas centradas no isocentro do plano axial
    x = (np.arange(nx) - (nx - 1) / 2.0) * sx
    y = (np.arange(ny) - (ny - 1) / 2.0) * sy

    xx, yy = np.meshgrid(x, y, indexing="xy")

    if radius_mm is None:
        radius_mm = min(nx * sx, ny * sy) / 2.0

    r = np.sqrt(xx**2 + yy**2)
    mask_inside = r <= radius_mm

    # Aplica em todas as fatias axiais simultaneamente
    arr[:, ~mask_inside] = air_hu

    return arr, radius_mm


# ──────────────────────────────────────────────────────────────────────────────
# Reorientação espacial (transposição + espelhamento)
# ──────────────────────────────────────────────────────────────────────────────

def reorient_to_ras(arr_zyx, flip_y=True):
    """
    Aplica a reorientação espacial necessária para alinhar o volume com
    o sistema de coordenadas anatômico RAS (Right-Anterior-Superior),
    esperado pelo NIfTI e pelas ferramentas de segmentação/simulação.

    As operações realizadas são:
      1. Transposição dos eixos: (z, y, x) → (x, y, z)   [np.transpose]
      2. Espelhamento no eixo Y: corrige a orientação anterior-posterior

    Parâmetros
    ----------
    arr_zyx : np.ndarray, shape (z, y, x)
        Volume em ordem numpy padrão (z, y, x).
    flip_y : bool
        Se True (padrão), aplica espelhamento no eixo Y após a transposição.

    Retorna
    -------
    arr_xyz : np.ndarray, shape (x, y, z)
        Volume reorientado, pronto para ser encapsulado em Nifti1Image.
    """
    # Passo 1 — transposição (z,y,x) → (x,y,z)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))

    # Passo 2 — espelhamento no eixo Y (anterior-posterior)
    if flip_y:
        arr_xyz = arr_xyz[:, ::-1, :]

    return arr_xyz


# ──────────────────────────────────────────────────────────────────────────────
# Função principal de conversão
# ──────────────────────────────────────────────────────────────────────────────

def convert_npy_to_nii(
    input_path,
    output_path=None,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    dtype="int16",
    array_order="zyx",
    center=False,
    flip_y=True,
    hu_slope=1000.0,
    hu_intercept=-1000.0,
    mask_cylinder=False,
    air_hu=-1000.0,
    cylinder_radius_mm=None,
    min_hu=None,
    max_hu=None,
):
    """
    Converte um volume .npy para o formato NIfTI (.nii.gz).

    O pipeline é composto pelas seguintes etapas sequenciais:
      1. Carregamento e validação do array 3D
      2. Ajuste da ordem dos eixos (zyx → zyx ou xyz → zyx)
      3. Calibração linear para escala HU
      4. Máscara cilíndrica (opcional)
      5. Limpeza de valores não físicos (opcional)
      6. Conversão de tipo (int16, float32, uint8)
      7. Reorientação para RAS (transposição + espelhamento Y)
      8. Construção da matriz affine e exportação NIfTI
    """

    input_path = Path(input_path)

    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_path}")
        sys.exit(1)

    if output_path is None:
        output_path = input_path.with_suffix("").with_suffix(".nii.gz")
    else:
        output_path = Path(output_path)
        # garante extensão correta
        if output_path.suffix == ".gz" and output_path.stem.endswith(".nii"):
            pass  # já correto
        elif output_path.suffix == ".nii":
            output_path = output_path.with_suffix(".nii.gz")
        else:
            output_path = Path(str(output_path) + ".nii.gz")

    # ── 1. Carregamento ────────────────────────────────────────────────────────
    print(f"Lendo: {input_path}")
    arr = np.load(input_path)

    if arr.ndim != 3:
        print(f"[ERRO] Esperava array 3D, mas encontrei shape {arr.shape}")
        sys.exit(1)

    print("\nArray original:")
    print(f"  Shape : {arr.shape}")
    print(f"  Dtype : {arr.dtype}")
    print(f"  Range : {arr.min():.4f} → {arr.max():.4f}")

    # ── 2. Ajuste da ordem dos eixos → zyx ────────────────────────────────────
    if array_order == "zyx":
        arr_zyx = arr
    elif array_order == "xyz":
        arr_zyx = np.transpose(arr, (2, 1, 0))
    else:
        print("[ERRO] array_order deve ser 'zyx' ou 'xyz'")
        sys.exit(1)

    # ── 3. Calibração HU ──────────────────────────────────────────────────────
    arr_hu = arr_zyx.astype(np.float32) * hu_slope + hu_intercept

    print("\nDepois da calibração HU:")
    print(f"  HU slope     : {hu_slope}")
    print(f"  HU intercept : {hu_intercept}")
    print(f"  Range        : {arr_hu.min():.2f} → {arr_hu.max():.2f}")

    # ── 4. Máscara cilíndrica (opcional) ──────────────────────────────────────
    if mask_cylinder:
        arr_hu, used_radius = apply_cylindrical_air_mask(
            arr_hu,
            spacing_xyz=spacing,
            air_hu=air_hu,
            radius_mm=cylinder_radius_mm,
        )
        print("\nMáscara cilíndrica aplicada:")
        print(f"  Raio usado       : {used_radius:.3f} mm")
        print(f"  HU fora cilindro : {air_hu}")

    # ── 5. Limpeza de valores não físicos (opcional) ──────────────────────────
    if min_hu is not None:
        n = int(np.sum(arr_hu < min_hu))
        arr_hu[arr_hu < min_hu] = air_hu
        print(f"\nCorreção inferior: {n} voxels < {min_hu} → {air_hu}")

    if max_hu is not None:
        n = int(np.sum(arr_hu > max_hu))
        arr_hu[arr_hu > max_hu] = max_hu
        print(f"Correção superior: {n} voxels > {max_hu} truncados em {max_hu}")

    # ── 6. Conversão de tipo ──────────────────────────────────────────────────
    if dtype == "int16":
        arr_out = np.round(arr_hu)
        arr_out = np.clip(arr_out, -32768, 32767)
        arr_out = arr_out.astype(np.int16)

    elif dtype == "float32":
        arr_out = arr_hu.astype(np.float32)

    elif dtype == "uint8":
        print("[AVISO] uint8 não é adequado para HU negativos.")
        arr_out = np.clip(arr_hu, 0, 255)
        arr_out = np.round(arr_out).astype(np.uint8)

    else:
        print("[ERRO] dtype deve ser int16, float32 ou uint8")
        sys.exit(1)

    # ── 7. Reorientação espacial → RAS ────────────────────────────────────────
    # NIfTI espera o array em ordem (x, y, z); a reorientação também aplica
    # o espelhamento Y para alinhar com o sistema anatômico RAS.
    arr_ras = reorient_to_ras(arr_out, flip_y=flip_y)

    print("\nDepois da reorientação (RAS):")
    print(f"  Shape (x,y,z) : {arr_ras.shape}")

    # ── 8. Matriz affine e exportação NIfTI ───────────────────────────────────
    sx, sy, sz = spacing

    if center:
        # Origem calculada para centralizar o volume no espaço de coordenadas
        nx, ny_dim, nz_dim = arr_ras.shape
        ox = -(nx * sx) / 2.0
        oy = -(ny_dim * sy) / 2.0
        oz = -(nz_dim * sz) / 2.0
        used_origin = (ox, oy, oz)
        print(f"\nOrigem centralizada calculada: {used_origin}")
    else:
        used_origin = origin

    ox, oy, oz = used_origin

    # Matriz affine 4×4: codifica espaçamento, origem e orientação RAS
    # A convenção NIfTI usa (x, y, z) com espaçamento na diagonal principal.
    affine = np.array([
        [sx,  0,  0, ox],
        [ 0, sy,  0, oy],
        [ 0,  0, sz, oz],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)

    img = nib.Nifti1Image(arr_ras, affine)

    # Metadados de cabeçalho NIfTI
    hdr = img.header
    hdr.set_zooms((sx, sy, sz))          # espaçamento em mm
    hdr.set_xyzt_units("mm")             # unidade espacial
    hdr["qform_code"] = 1                # scanner coordinates
    hdr["sform_code"] = 1
    img.set_qform(affine)
    img.set_sform(affine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))

    print("\nSalvo em:")
    print(f"  {output_path}")
    print("\nImagem final:")
    print(f"  Shape   (x,y,z) : {arr_ras.shape}")
    print(f"  Spacing (x,y,z) : ({sx}, {sy}, {sz}) mm")
    print(f"  Origin  (x,y,z) : {used_origin}")
    print(f"  Dtype           : {arr_out.dtype}")
    print(f"  Range           : {arr_out.min()} → {arr_out.max()}")
    print(f"  Flip Y aplicado : {flip_y}")


# ──────────────────────────────────────────────────────────────────────────────
# Interface de linha de comando
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converte volume .npy para NIfTI (.nii.gz) com calibração HU "
                    "e reorientação espacial.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        help="Arquivo .npy de entrada",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Arquivo .nii.gz de saída (padrão: mesmo nome do input)",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Espaçamento dos voxels em mm, ordem x y z (padrão: 1 1 1)",
    )

    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("OX", "OY", "OZ"),
        help="Origem da imagem em mm, ordem x y z (padrão: 0 0 0)",
    )

    parser.add_argument(
        "--center",
        action="store_true",
        help="Centraliza o volume na origem (ignora --origin)",
    )

    parser.add_argument(
        "--dtype",
        default="int16",
        choices=["int16", "float32", "uint8"],
        help="Tipo de dado de saída (padrão: int16)",
    )

    parser.add_argument(
        "--array-order",
        default="zyx",
        choices=["zyx", "xyz"],
        help="Ordem dos eixos no .npy (padrão: zyx)",
    )

    parser.add_argument(
        "--hu-slope",
        type=float,
        default=1000.0,
        help="Coeficiente multiplicativo da calibração HU (padrão: 1000.0)",
    )

    parser.add_argument(
        "--hu-intercept",
        type=float,
        default=-1000.0,
        help="Intercepto da calibração HU (padrão: -1000.0)",
    )

    parser.add_argument(
        "--mask-cylinder",
        action="store_true",
        help="Aplica máscara cilíndrica no plano x-y, preenchendo exterior com --air-hu",
    )

    parser.add_argument(
        "--air-hu",
        type=float,
        default=-1000.0,
        help="Valor de HU atribuído ao ar / região externa ao cilindro (padrão: -1000.0)",
    )

    parser.add_argument(
        "--cylinder-radius-mm",
        type=float,
        default=None,
        help="Raio do cilindro em mm (padrão: min(Nx*sx, Ny*sy)/2)",
    )

    parser.add_argument(
        "--min-hu",
        type=float,
        default=None,
        help="Voxels abaixo deste HU são substituídos por --air-hu",
    )

    parser.add_argument(
        "--max-hu",
        type=float,
        default=None,
        help="Voxels acima deste HU são truncados neste valor",
    )

    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Desativa o espelhamento no eixo Y durante a reorientação",
    )

    args = parser.parse_args()

    convert_npy_to_nii(
        input_path=args.input,
        output_path=args.output,
        spacing=tuple(args.spacing),
        origin=tuple(args.origin),
        dtype=args.dtype,
        array_order=args.array_order,
        center=args.center,
        flip_y=not args.no_flip_y,
        hu_slope=args.hu_slope,
        hu_intercept=args.hu_intercept,
        mask_cylinder=args.mask_cylinder,
        air_hu=args.air_hu,
        cylinder_radius_mm=args.cylinder_radius_mm,
        min_hu=args.min_hu,
        max_hu=args.max_hu,
    )
