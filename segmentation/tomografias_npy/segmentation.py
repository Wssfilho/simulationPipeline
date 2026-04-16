import os
from multiprocessing import freeze_support

from totalsegmentator.python_api import totalsegmentator


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "imagens_ct")
    output_base_dir = os.path.join(base_dir, "resultados")

    os.makedirs(output_base_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Pasta de entrada não encontrada: {input_dir}")

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".nii.gz")])

    if not files:
        print("Nenhum arquivo .nii.gz encontrado em imagens_ct.")
        return

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(
            output_base_dir, filename.replace(".nii.gz", ""))

        print(f"Processando {filename}...")

        totalsegmentator(
            input=input_path,
            output=output_path,
            task="total",
            ml=True,
            # precisa-se testar em um computador com GPU, o modo fast false não execultou as imagens corretamente

            fast=True,
            nr_thr_resamp=1,
            nr_thr_saving=1,
            device="cpu",
        )

    print("Tarefa concluída!")


if __name__ == "__main__":
    freeze_support()
    main()
