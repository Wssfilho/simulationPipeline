from pathlib import Path
import SimpleITK as sitk


def sum_mhd_series(folder, prefix, output_name, pixel_type=sitk.sitkFloat64):
    """
    Soma todos os arquivos prefix-*.mhd dentro de uma pasta
    e salva o resultado em output_name.mhd/raw.

    Exemplo:
        prefix = "Dose"
        encontra Dose-*.mhd
        salva Dose-total.mhd/raw
    """

    folder = Path(folder)

    files = sorted(folder.glob(f"{prefix}-*.mhd"))

    # Evita pegar arquivos ja somados, se existirem
    files = [f for f in files if "total" not in f.stem.lower()]

    if len(files) == 0:
        print(f"Nenhum arquivo encontrado para {prefix}-*.mhd")
        return

    print(f"\nSomando {len(files)} arquivos de {prefix}:")
    for f in files:
        print("  ", f.name)

    accumulator = None
    reference = None

    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] lendo {f.name}")

        img = sitk.ReadImage(str(f), pixel_type)

        if accumulator is None:
            accumulator = img
            reference = img
        else:
            # Checagem simples de compatibilidade
            if img.GetSize() != reference.GetSize():
                raise ValueError(f"Tamanho diferente em {f.name}")

            if img.GetSpacing() != reference.GetSpacing():
                raise ValueError(f"Spacing diferente em {f.name}")

            if img.GetOrigin() != reference.GetOrigin():
                raise ValueError(f"Origin diferente em {f.name}")

            accumulator = accumulator + img

        # Libera referencia da imagem atual
        del img

    output_path = folder / output_name

    print(f"Salvando: {output_path}")
    sitk.WriteImage(accumulator, str(output_path), useCompression=False)

    print("Concluido.")


# =========================
# Uso
# =========================

folder = "output"

sum_mhd_series(
    folder=folder,
    prefix="Dose",
    output_name="Dose-total.mhd"
)

sum_mhd_series(
    folder=folder,
    prefix="Edep",
    output_name="Edep-total.mhd"
)