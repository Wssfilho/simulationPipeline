"""
Pipeline: Segmentação NIfTI + Dose MHD → DVH por Órgão
========================================================
Etapas:
  1. Carregar arquivo .nii de segmentação (multi-label)
  2. Extrair máscara binária de cada órgão
  3. Carregar arquivo de dose .mhd (SimpleITK)
  4. Multiplicar máscara × dose → dose por órgão
  5. Calcular média das doses (voxels ≠ 0)
  6. Gerar DVH (Dose-Volume Histogram) cumulativo
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use("Agg")


# ── Mapeamento padrão de labels TotalSegmentator ──────────────────────────────
DEFAULT_LABEL_MAP = {
    1:  "Baço",
    2:  "Rim Direito",
    3:  "Rim Esquerdo",
    4:  "Vesícula Biliar",
    5:  "Fígado",
    6:  "Estômago",
    7:  "Aorta",
    8:  "Veia Cava Inferior",
    9:  "Pâncreas",
    10: "Glândula Adrenal Direita",
    11: "Glândula Adrenal Esquerda",
    12: "Pulmão Direito",
    13: "Pulmão Esquerdo",
    14: "Coração",
    15: "Esôfago",
    16: "Traqueia",
    17: "Próstata / Útero",
    18: "Bexiga",
    19: "Intestino Delgado",
    20: "Duodeno",

    # Adicione mais labels conforme seu arquivo
}


# ── 1. Carregar segmentação NIfTI ─────────────────────────────────────────────
def load_segmentation(nii_path: str):
    seg_img = sitk.ReadImage(nii_path)
    seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int32)  # (Z, Y, X)
    spacing = seg_img.GetSpacing()  # (X, Y, Z)
    labels = sorted(np.unique(seg_arr[seg_arr > 0]).tolist())
    print(
        f"[NIfTI] Shape: {seg_arr.shape} | Spacing: {spacing} mm | Labels: {labels}")
    return seg_img, seg_arr, spacing, labels


# ── 2. Carregar dose MHD ──────────────────────────────────────────────────────
def load_dose_mhd(mhd_path: str):
    dose_img = sitk.ReadImage(mhd_path)
    dose_arr = sitk.GetArrayFromImage(dose_img)          # shape: (Z, Y, X)
    dose_arr = dose_arr.astype(np.float32)
    spacing = dose_img.GetSpacing()                    # (X, Y, Z)
    print(f"[MHD]   Shape: {dose_arr.shape} | Spacing: {spacing} mm | "
          f"Dose range: [{dose_arr.min():.3f}, {dose_arr.max():.3f}]")
    return dose_img, dose_arr, spacing


# ── 3. Re-amostrar dose para o grid da segmentação ───────────────────────────
def resample_dose_to_seg(dose_img, seg_img):
    """
    Reamostra a dose para o grid da segmentacao usando SimpleITK,
    respeitando spacing/origin/direction do NIfTI.
    """
    same_grid = (
        dose_img.GetSize() == seg_img.GetSize()
        and np.allclose(dose_img.GetSpacing(), seg_img.GetSpacing())
        and np.allclose(dose_img.GetOrigin(), seg_img.GetOrigin())
        and np.allclose(dose_img.GetDirection(), seg_img.GetDirection())
    )

    if same_grid:
        print("[Resample] Grids ja coincidem, sem reamostragem.")
        return sitk.GetArrayFromImage(dose_img).astype(np.float32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(seg_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)

    dose_resampled_img = resampler.Execute(dose_img)
    dose_resampled = sitk.GetArrayFromImage(
        dose_resampled_img).astype(np.float32)
    print(
        f"[Resample] Dose reamostrada: {dose_img.GetSize()} → {dose_resampled_img.GetSize()}"
    )
    return dose_resampled


# ── 4. Extrair máscaras + calcular dose por órgão ─────────────────────────────
def compute_organ_doses(seg_zyx, dose_zyx, label_map):
    """
    seg_zyx : (Z, Y, X) – SimpleITK
    dose_zyx : (Z, Y, X) – SimpleITK
    Retorna dict: { label: { 'name', 'mask_binary', 'dose_voxels', 'mean_dose' } }
    """

    results = {}
    labels_present = sorted(np.unique(seg_zyx[seg_zyx > 0]).tolist())

    for lbl in labels_present:
        name = label_map.get(int(lbl), f"Orgão_{int(lbl)}")

        # Máscara binária
        mask_bin = (seg_zyx == lbl).astype(np.uint8)

        # Dose no órgão (multiplicação elemento a elemento)
        dose_organ = dose_zyx * mask_bin

        # Voxels com dose > 0
        dose_nonzero = dose_organ[dose_organ > 0]
        mean_dose = float(dose_nonzero.mean()
                          ) if dose_nonzero.size > 0 else 0.0
        volume_vox = int(mask_bin.sum())

        results[int(lbl)] = {
            "name":        name,
            "mask_binary": mask_bin,
            "dose_voxels": dose_organ,
            "dose_nonzero": dose_nonzero,
            "mean_dose":   mean_dose,
            "volume_vox":  volume_vox,
        }
        print(f"  Label {lbl:3d} | {name:<28} | Vol: {volume_vox:>8} vox | "
              f"Dose média: {mean_dose:.4f} Gy")

    return results


# ── 5. Gerar DVH cumulativo ───────────────────────────────────────────────────
def generate_dvh(organ_results, output_path="dvh.png", dose_unit="Gy"):
    n = len(organ_results)
    if n == 0:
        print("[DVH] Nenhum órgão encontrado.")
        return

    colors = cm.tab20(np.linspace(0, 1, n))
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax_dvh, ax_bar = axes

    dose_max_global = max(
        r["dose_nonzero"].max() if r["dose_nonzero"].size > 0 else 0
        for r in organ_results.values()
    )

    # ── DVH cumulativo ──────────────────────────────────────────────────────
    for (lbl, info), color in zip(organ_results.items(), colors):
        dose_vals = info["dose_nonzero"]
        if dose_vals.size == 0:
            continue

        bins = np.linspace(0, dose_max_global * 1.05, 300)
        counts, edges = np.histogram(dose_vals, bins=bins)
        cum_vol = 100.0 * (1 - np.cumsum(counts) / dose_vals.size)   # %

        ax_dvh.plot(
            edges[:-1], cum_vol,
            label=f"{info['name']} (D̄={info['mean_dose']:.2f} {dose_unit})",
            color=color, linewidth=1.8
        )

    ax_dvh.set_xlabel(f"Dose ({dose_unit})", fontsize=12)
    ax_dvh.set_ylabel("Volume (%)", fontsize=12)
    ax_dvh.set_title("DVH Cumulativo por Órgão",
                     fontsize=13, fontweight="bold")
    ax_dvh.set_ylim(0, 105)
    ax_dvh.set_xlim(0, dose_max_global * 1.05)
    ax_dvh.grid(True, alpha=0.3)
    ax_dvh.legend(fontsize=7.5, loc="upper right", framealpha=0.8)

    # ── Barras de dose média ────────────────────────────────────────────────
    names = [info["name"] for info in organ_results.values()]
    mean_doses = [info["mean_dose"] for info in organ_results.values()]

    bars = ax_bar.barh(names, mean_doses,
                       color=colors[:n], edgecolor="white", height=0.7)

    for bar, val in zip(bars, mean_doses):
        ax_bar.text(
            bar.get_width() + dose_max_global * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=8
        )

    ax_bar.set_xlabel(f"Dose Média ({dose_unit})", fontsize=12)
    ax_bar.set_title("Dose Média por Órgão\n(voxels com dose > 0)",
                     fontsize=13, fontweight="bold")
    ax_bar.set_xlim(0, dose_max_global * 1.15)
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.invert_yaxis()

    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[DVH] Gráfico salvo em: {output_path}")
    return output_path


# ── 6. Exportar máscaras binárias como NIfTI ─────────────────────────────────
def export_masks(organ_results, ref_img, out_dir="organ_masks"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for lbl, info in organ_results.items():
        mask_zyx = info["mask_binary"].astype(np.uint8)
        safe_name = info["name"].replace(" ", "_").replace("/", "-")
        fname = Path(out_dir) / f"mask_{lbl:03d}_{safe_name}.nii.gz"
        mask_img = sitk.GetImageFromArray(mask_zyx)
        mask_img.CopyInformation(ref_img)
        sitk.WriteImage(mask_img, str(fname))
        print(f"  Salvo: {fname}")

    print(f"[Máscaras] {len(organ_results)} arquivo(s) em '{out_dir}/'")


# ── 7. Relatório CSV ──────────────────────────────────────────────────────────
def export_csv(organ_results, out_csv="dose_summary.csv", dose_unit="Gy"):
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Label", "Orgão", "Volume (voxels)", f"Dose Média ({dose_unit})",
                    f"D_min ({dose_unit})", f"D_max ({dose_unit})", f"D50 ({dose_unit})"])
        for lbl, info in organ_results.items():
            dv = info["dose_nonzero"]
            if dv.size > 0:
                d_min, d_max, d50 = dv.min(), dv.max(), np.median(dv)
            else:
                d_min = d_max = d50 = 0.0
            w.writerow([lbl, info["name"], info["volume_vox"],
                        f"{info['mean_dose']:.6f}", f"{d_min:.6f}",
                        f"{d_max:.6f}", f"{d50:.6f}"])
    print(f"[CSV] Relatório salvo em: {out_csv}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: NIfTI segmentação + MHD dose → DVH por órgão"
    )
    parser.add_argument("--seg",      required=True,
                        help="Arquivo .nii/.nii.gz de segmentação")
    parser.add_argument("--dose",     required=True,
                        help="Arquivo .mhd de dose")
    parser.add_argument("--out_dvh",  default="dvh_organs.png",
                        help="Saída do gráfico DVH")
    parser.add_argument(
        "--out_csv",  default="dose_summary.csv", help="Saída CSV com resumo")
    parser.add_argument("--out_masks", default="organ_masks",
                        help="Pasta para máscaras NIfTI")
    parser.add_argument("--dose_unit", default="Gy",
                        help="Unidade de dose (Gy, cGy…)")
    parser.add_argument("--no_masks", action="store_true",
                        help="Não exportar máscaras NIfTI")
    args = parser.parse_args()

    print("\n═══════════════════════════════════════════════════")
    print("  Pipeline: Segmentação → Dose → DVH")
    print("═══════════════════════════════════════════════════\n")

    # 1. Segmentação
    seg_img, seg_zyx, _seg_spacing, labels = load_segmentation(args.seg)

    # 2. Dose
    dose_img, _dose_arr, _dose_spacing = load_dose_mhd(args.dose)

    # 3. Re-amostrar dose
    dose_resampled = resample_dose_to_seg(dose_img, seg_img)

    # 4. Máscaras + dose por órgão
    print("\n[Órgãos] Calculando doses por órgão…")
    organ_results = compute_organ_doses(
        seg_zyx, dose_resampled, DEFAULT_LABEL_MAP)

    # 5. DVH
    dvh_path = generate_dvh(
        organ_results, output_path=args.out_dvh, dose_unit=args.dose_unit)

    # 6. Exportar máscaras (opcional)
    if not args.no_masks:
        print(f"\n[Máscaras] Exportando…")
        export_masks(organ_results, seg_img, out_dir=args.out_masks)

    # 7. CSV
    export_csv(organ_results, out_csv=args.out_csv, dose_unit=args.dose_unit)

    print("\n═══════════════════════════════════════════════════")
    print("  Concluído com sucesso!")
    print("═══════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
