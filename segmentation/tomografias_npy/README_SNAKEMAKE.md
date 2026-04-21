# Pipeline com Snakemake

Este pipeline automatiza:

1. Conversao de arquivos `.npy` para `.nii.gz`
2. Segmentacao com TotalSegmentator

## Estrutura esperada

- Entradas `.npy`: pasta atual (`segmentation/tomografias_npy/`)
- Entradas `.nii.gz` ja prontas: `imagens_ct/`
- Saidas da segmentacao: `resultados/<sample>/`

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
```

## Observacoes

- Se existir `imagens_ct/<sample>.nii.gz`, o Snakemake usa esse arquivo diretamente.
- Se nao existir, mas houver `<sample>.npy`, o arquivo sera convertido antes da segmentacao.
- O script `segmentation_automatized.py` e usado na etapa de segmentacao.
