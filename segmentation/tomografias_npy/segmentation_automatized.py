import argparse
from multiprocessing import freeze_support
from pathlib import Path
from npy_to_nii_calibrated import npy_to_nii
from totalsegmentator.python_api import totalsegmentator


def run_totalsegmentator(input_path: Path, output_base_dir: Path) -> None:
    output_path = output_base_dir / input_path.name.replace(".nii.gz", "")

    print(f"Processando {input_path.name}...")
    totalsegmentator(
        input=str(input_path),
        output=str(output_path),
        task="total",
        ml=True,
        # Em CPU, fast=True evitou falhas de execução neste projeto.
        fast=True,
        nr_thr_resamp=1,
        nr_thr_saving=1,
        device="cpu",
    )


def process_directory(input_dir: Path, output_base_dir: Path) -> None:
    files = sorted(input_dir.glob("*.nii.gz"))

    if not files:
        print("Nenhum arquivo .nii.gz encontrado em imagens_ct.")
        return

    for input_path in files:
        run_totalsegmentator(input_path, output_base_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline: converte .npy para .nii.gz e executa segmentação"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Arquivo de entrada (.npy ou .nii.gz). "
            "Se omitido, processa todos .nii.gz da pasta imagens_ct/."
        ),
    )
    parser.add_argument(
        "--spacing",
        "-s",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
        help="Spacing usado na conversão do .npy (padrão: 1.0 1.0 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "imagens_ct"
    output_base_dir = base_dir / "resultados"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input).expanduser()
        if not input_path.is_absolute():
            input_path = (Path.cwd() / input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

        if input_path.suffix.lower() == ".npy":
            input_dir.mkdir(parents=True, exist_ok=True)
            converted_path = input_dir / f"{input_path.stem}.nii.gz"
            print(f"Convertendo {input_path.name} para NIfTI...")
            npy_to_nii(
                input_path=str(input_path),
                output_path=str(converted_path),
                spacing=tuple(args.spacing),
            )
            run_totalsegmentator(converted_path, output_base_dir)
        elif input_path.name.lower().endswith(".nii.gz"):
            run_totalsegmentator(input_path, output_base_dir)
        else:
            raise ValueError("Formato inválido. Use .npy ou .nii.gz")

        print("Tarefa concluída!")
        return

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Pasta de entrada não encontrada: {input_dir}")

    process_directory(input_dir, output_base_dir)
    print("Tarefa concluída!")


if __name__ == "__main__":
    freeze_support()
    main()
