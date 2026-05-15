#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converte volume .npy para .mhd + .raw.

Assume por padrão que o .npy está em ordem (z, y, x),
que é a ordem usual quando vem de SimpleITK/GetArrayFromImage.

Exemplos:

python npy_to_mhd.py volume.npy --output volume.mhd

python npy_to_mhd.py volume.npy --output volume.mhd \
    --spacing 1 1 1 \
    --origin -128 -128 -128 \
    --dtype int16

python npy_to_mhd.py volume.npy --output volume.mhd \
    --array-order zyx \
    --mask-cylinder \
    --air-hu -1000 \
    --min-hu -1200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def apply_cylindrical_air_mask(arr, spacing_xyz, air_hu=-1000.0, radius_mm=None):
    """
    Aplica cilindro no plano x-y.

    arr deve estar em ordem numpy: (z, y, x)
    spacing_xyz deve estar em ordem física: (sx, sy, sz)
    """

    nz, ny, nx = arr.shape
    sx, sy, sz = spacing_xyz

    x = (np.arange(nx) - (nx - 1) / 2.0) * sx
    y = (np.arange(ny) - (ny - 1) / 2.0) * sy

    xx, yy = np.meshgrid(x, y, indexing="xy")

    if radius_mm is None:
        radius_mm = min(nx * sx, ny * sy) / 2.0

    r = np.sqrt(xx**2 + yy**2)
    mask_inside = r <= radius_mm

    arr[:, ~mask_inside] = air_hu

    return arr, radius_mm


def convert_npy_to_mhd(
    input_path,
    output_path=None,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    dtype="int16",
    array_order="zyx",
    center=False,
    hu_slope=1000.0,
    hu_intercept=-1000.0,
    mask_cylinder=True,
    air_hu=-1000.0,
    cylinder_radius_mm=None,
    min_hu=None,
    max_hu=None,
):

    input_path = Path(input_path)

    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_path}")
        sys.exit(1)

    if output_path is None:
        output_path = input_path.with_suffix(".mhd")
    else:
        output_path = Path(output_path)

    print(f"Lendo: {input_path}")

    arr = np.load(input_path)

    if arr.ndim != 3:
        print(f"[ERRO] Esperava array 3D, mas encontrei shape {arr.shape}")
        sys.exit(1)

    print("\nArray original:")
    print(f"  Shape : {arr.shape}")
    print(f"  Dtype : {arr.dtype}")
    print(f"  Range : {arr.min()} → {arr.max()}")

    # --------------------------------------------------
    # Ajuste da ordem dos eixos
    # --------------------------------------------------
    if array_order == "zyx":
        # já está no formato esperado pelo SimpleITK GetImageFromArray
        arr_zyx = arr

    elif array_order == "xyz":
        # converte de (x, y, z) para (z, y, x)
        arr_zyx = np.transpose(arr, (2, 1, 0))

    else:
        print("[ERRO] array_order deve ser 'zyx' ou 'xyz'")
        sys.exit(1)

    arr_hu = arr_zyx.astype(np.float32) * hu_slope + hu_intercept

    print("\nDepois da escala:")
    print(f"  HU slope     : {hu_slope}")
    print(f"  HU intercept : {hu_intercept}")
    print(f"  Range        : {arr_hu.min()} → {arr_hu.max()}")

    # --------------------------------------------------
    # Máscara cilíndrica externa
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Limpeza de valores não físicos
    # --------------------------------------------------
    if min_hu is not None:
        n = np.sum(arr_hu < min_hu)
        arr_hu[arr_hu < min_hu] = air_hu

        print("\nCorreção inferior:")
        print(f"  Valores < {min_hu} colocados em {air_hu}")
        print(f"  Voxels corrigidos: {n}")

    if max_hu is not None:
        n = np.sum(arr_hu > max_hu)
        arr_hu[arr_hu > max_hu] = max_hu

        print("\nCorreção superior:")
        print(f"  Valores > {max_hu} truncados")
        print(f"  Voxels corrigidos: {n}")

    # --------------------------------------------------
    # Conversão de tipo
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Cria imagem SimpleITK
    # --------------------------------------------------
    img = sitk.GetImageFromArray(arr_out)

    # SimpleITK usa spacing/origin em ordem (x, y, z)
    img.SetSpacing(tuple(spacing))

    if center:
        size_xyz = np.array(img.GetSize())
        spacing_xyz = np.array(img.GetSpacing())
        new_origin = -(size_xyz * spacing_xyz) / 2.0
        img.SetOrigin(tuple(new_origin))
    else:
        img.SetOrigin(tuple(origin))

    # --------------------------------------------------
    # Salva MHD/RAW
    # --------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))

    print("\nSalvo em:")
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.raw')}")

    print("\nImagem final:")
    print(f"  Size    : {img.GetSize()}")       # x, y, z
    print(f"  Spacing : {img.GetSpacing()}")    # x, y, z
    print(f"  Origin  : {img.GetOrigin()}")
    print(f"  Dtype   : {img.GetPixelIDTypeAsString()}")
    print(f"  Range   : {arr_out.min()} → {arr_out.max()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converte volume .npy para .mhd + .raw"
    )

    parser.add_argument("input", help="Arquivo .npy de entrada")

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Arquivo .mhd de saída"
    )

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Espaçamento dos voxels em mm, ordem x y z"
    )

    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("OX", "OY", "OZ"),
        help="Origem da imagem, ordem x y z"
    )

    parser.add_argument(
        "--center",
        action="store_true",
        help="Centraliza a imagem na origem"
    )

    parser.add_argument(
        "--dtype",
        default="int16",
        choices=["int16", "float32", "uint8"],
        help="Tipo de saída"
    )

    parser.add_argument(
        "--array-order",
        default="zyx",
        choices=["zyx", "xyz"],
        help="Ordem dos eixos no arquivo .npy. Padrão: zyx"
    )

    parser.add_argument(
        "--hu-slope",
        type=float,
        default=1000.0,
        help="Fator multiplicativo para HU"
    )

    parser.add_argument(
        "--hu-intercept",
        type=float,
        default=-1000.0,
        help="Intercepto para HU"
    )

    parser.add_argument(
        "--mask-cylinder",
        action="store_true",
        help="Coloca fora do cilindro com HU do ar"
    )

    parser.add_argument(
        "--air-hu",
        type=float,
        default=-1000.0,
        help="HU usado para ar/fundo"
    )

    parser.add_argument(
        "--cylinder-radius-mm",
        type=float,
        default=None,
        help="Raio do cilindro em mm. Se omitido, usa arista/2"
    )

    parser.add_argument(
        "--min-hu",
        type=float,
        default=None,
        help="Valores abaixo deste limite são colocados em --air-hu"
    )

    parser.add_argument(
        "--max-hu",
        type=float,
        default=None,
        help="Valores acima deste limite são truncados"
    )

    args = parser.parse_args()

    convert_npy_to_mhd(
        input_path=args.input,
        output_path=args.output,
        spacing=tuple(args.spacing),
        origin=tuple(args.origin),
        dtype=args.dtype,
        array_order=args.array_order,
        center=args.center,
        hu_slope=args.hu_slope,
        hu_intercept=args.hu_intercept,
        mask_cylinder=True,
        air_hu=args.air_hu,
        cylinder_radius_mm=args.cylinder_radius_mm,
        min_hu=args.min_hu,
        max_hu=args.max_hu,
    )