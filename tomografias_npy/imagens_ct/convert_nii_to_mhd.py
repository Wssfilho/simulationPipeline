#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converte imagens NIfTI (.nii ou .nii.gz) para o par .mhd + .raw
compatível com OpenGate/SimpleITK.

Uso:
    python convert_nii_to_mhd.py entrada.nii.gz
    python convert_nii_to_mhd.py entrada.nii.gz --output saida.mhd
    python convert_nii_to_mhd.py entrada.nii.gz --center   # centraliza na origem
    python convert_nii_to_mhd.py entrada.nii.gz --dtype int16
"""

import argparse
import sys
from pathlib import Path
import SimpleITK as sitk
import numpy as np


def convert(input_path: str,
            output_path: str | None = None,
            center: bool = False,
            dtype: str = "int16") -> None:

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_path}")
        sys.exit(1)

    # --- saída padrão: mesmo nome, extensão .mhd ---
    if output_path is None:
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        output_path = input_path.parent / (stem + ".mhd")
    else:
        output_path = Path(output_path)

    print(f"Lendo:   {input_path}")
    img = sitk.ReadImage(str(input_path))

    # --- info antes ---
    size = np.array(img.GetSize())       # (x, y, z)
    spacing = np.array(img.GetSpacing())    # mm
    origin = np.array(img.GetOrigin())
    arr = sitk.GetArrayFromImage(img)   # shape (z, y, x) no numpy

    print(f"  Tamanho  : {img.GetSize()}")
    print(f"  Spacing  : {img.GetSpacing()} mm")
    print(f"  Origin   : {img.GetOrigin()}")
    print(f"  Dtype    : {img.GetPixelIDTypeAsString()}")
    print(f"  HU range : {arr.min()} → {arr.max()}")

    # --- cast para o tipo desejado ---
    dtype_map = {
        "int16":   sitk.sitkInt16,
        "float32": sitk.sitkFloat32,
        "uint8":   sitk.sitkUInt8,
    }
    if dtype not in dtype_map:
        print(f"[ERRO] dtype deve ser: {list(dtype_map.keys())}")
        sys.exit(1)
    img = sitk.Cast(img, dtype_map[dtype])

    # --- centralizar na origem (recomendado para OpenGate) ---
    if center:
        new_origin = -(size * spacing) / 2.0
        img.SetOrigin(new_origin.tolist())
        print(f"  Origin → : {img.GetOrigin()}  (centralizado)")

    # --- salvar .mhd + .raw ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
    raw_path = output_path.with_suffix(".raw")

    print(f"\nSalvo em : {output_path}")
    print(f"           {raw_path}")
    print(f"  Tamanho final: {img.GetSize()}")
    print(f"  Origin  final: {img.GetOrigin()}")
    print("Conversão concluída.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte .nii/.nii.gz → .mhd + .raw para OpenGate"
    )
    parser.add_argument("input",
                        help="Arquivo de entrada (.nii ou .nii.gz)")
    parser.add_argument("--output", "-o", default=None,
                        help="Arquivo de saída (.mhd) [padrão: mesmo nome]")
    parser.add_argument("--center", action="store_true",
                        help="Centraliza a imagem na origem (0,0,0)")
    parser.add_argument("--dtype", default="int16",
                        choices=["int16", "float32", "uint8"],
                        help="Tipo de dado de saída (padrão: int16)")
    args = parser.parse_args()

    convert(args.input, args.output, args.center, args.dtype)
