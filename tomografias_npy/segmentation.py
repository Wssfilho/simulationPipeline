import argparse
from multiprocessing import freeze_support
from pathlib import Path

from totalsegmentator.python_api import totalsegmentator


def sample_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError("Arquivo de entrada deve terminar em .nii ou .nii.gz")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except Exception:
        return "cpu"

    return "gpu" if torch.cuda.is_available() else "cpu"


def run_totalsegmentator(input_path: Path, output_path: Path, device: str) -> None:
    print(f"Processando {input_path.name} -> {output_path.name}...")
    resolved_device = resolve_device(device)
    totalsegmentator(
        input=str(input_path),
        output=str(output_path),
        task="total",
        ml=True,
        # precisa-se testar em um computador com GPU, o modo fast false nao executou as imagens corretamente
        fast=True,
        nr_thr_resamp=1,
        nr_thr_saving=1,
        device=resolved_device,
        statistics=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa segmentacao com TotalSegmentator"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Arquivo .nii ou .nii.gz para segmentar. "
            "Se omitido, processa todos os .nii.gz de imagens_ct/."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Arquivo de saida .nii para um processamento unico.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Dispositivo de execucao (padrao: auto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "imagens_ct"
    output_base_dir = base_dir / "resultados"

    output_base_dir.mkdir(parents=True, exist_ok=True)

    if args.output and not args.input:
        raise ValueError("--output requer um arquivo de entrada.")

    if args.input:
        input_path = Path(args.input).expanduser()
        if not input_path.is_absolute():
            input_path = (Path.cwd() / input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(
                f"Arquivo de entrada nao encontrado: {input_path}")

        if args.output:
            output_path = Path(args.output).expanduser()
            if not output_path.is_absolute():
                output_path = (Path.cwd() / output_path).resolve()
        else:
            sample_name = sample_name_from_path(input_path)
            output_path = output_base_dir / f"{sample_name}.nii"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_totalsegmentator(input_path, output_path, args.device)
        print("Tarefa concluida!")
        return

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Pasta de entrada nao encontrada: {input_dir}")

    files = sorted(input_dir.glob("*.nii.gz"))

    if not files:
        print("Nenhum arquivo .nii.gz encontrado em imagens_ct.")
        return

    for input_path in files:
        sample_name = sample_name_from_path(input_path)
        output_path = output_base_dir / f"{sample_name}.nii"
        run_totalsegmentator(input_path, output_path, args.device)

    print("Tarefa concluida!")


if __name__ == "__main__":
    freeze_support()
    main()
