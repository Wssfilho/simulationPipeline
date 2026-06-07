# Pipeline com Snakemake

Este pipeline automatiza:

1. Conversao de arquivos `.npy` para `.nii.gz`
2. Segmentacao com TotalSegmentator

# caso de erro na instalacao

Traceback (most recent call last):

File "/home/wilson/tcc/venv/bin/snakemake", line 6, in <module>

sys.exit(main())

File "/home/wilson/tcc/venv/lib/python3.10/site-packages/snakemake/**init**.py", line 2736, in main

parser = get_argument_parser()

File "/home/wilson/tcc/venv/lib/python3.10/site-packages/snakemake/**init**.py", line 1607, in get_argument_parser

lp_solvers = pulp.list_solvers(onlyAvailable=True)

AttributeError: module 'pulp' has no attribute 'list_solvers'. Did you mean: 'listSolvers'?

pip install "pulp<2.8.0"

pip install --upgrade snakemake

## Estrutura esperada

- Entradas `.npy`: pasta atual (`segmentation/tomografias_npy/`)
- Entradas `.nii.gz` ja prontas: `imagens_ct/`
- Saidas da segmentacao: `resultados/<sample>.nii`

## Arquivos do workflow

- `Snakefile`
- `config.yaml`

## Parametros

Edite o arquivo `config.yaml` para ajustar o spacing da conversao:

```yaml
spacing:
  - 1.0
  - 1.0
  - 1.0
```

## Execucao

No diretorio `segmentation/tomografias_npy/`:

```bash
snakemake -n -p
```

Com execucao real (exemplo com 1 core):

```bash
snakemake --cores 1 -p
#execucao com 6 cores
snakemake --cores 6 -p
```

## Observacoes

- Se existir `imagens_ct/<sample>.nii.gz`, o Snakemake usa esse arquivo diretamente.
- Se nao existir, mas houver `<sample>.npy`, o arquivo sera convertido antes da segmentacao.
- A etapa de segmentacao usa `segmentation.py` com contrato 1:1 (uma entrada para uma saida).

## DAG resumido

Para ficar mais legivel em apresentacao ou relatorio, a melhor opcao e mostrar o fluxo da esquerda para a direita e numerar as etapas principais. No caso deste pipeline, o DAG resumido fica assim:

```mermaid
flowchart LR
  A[1. Entrada: .npy ou .nii.gz] --> B[2. Conversao .npy -> .nii.gz]
  A --> C[3. Segmentacao]
  B --> C
  C --> D[4. Simulacao CBCT por angulo\n(jobs paralelos)]
  D --> E[5. Soma dos resultados]
  E --> F[6. Estatisticas de dose e DVH]
```

Se quiser gerar a figura do Snakemake em arquivo, use:

```bash
snakemake --dag -s Snakefile | dot -Grankdir=LR -Tpng > dag_lr.png
snakemake --dag -s Snakefile | dot -Grankdir=LR -Tsvg > dag_lr.svg
```
