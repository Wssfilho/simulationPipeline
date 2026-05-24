# Execucoes manuais (sem Snakemake)

Este README mostra como rodar cada script individualmente a partir da pasta
`tomografias_npy/`.

> Dica: execute todos os comandos a partir desta pasta:
>
> ```bash
> cd /home/wilson/tcc/tomografias_npy
> ```

## 1) NPY -> NIfTI (.nii.gz)

Script: `npy2nii.py`

```bash
python3 npy2nii.py 0801_clean_fdk_256.npy \
  --output imagens_ct/0801_clean_fdk_256.nii.gz \
  --spacing 1.5 1.5 1.5 \
  --center
```

Saidas:

- `imagens_ct/0801_clean_fdk_256.nii.gz`

Observacoes:

- Use `--lps` se precisar do NIfTI em LPS (compatibilidade ITK/vv).
- Sem `--output`, o arquivo sai com o mesmo nome do input.

## 2) Segmentacao (TotalSegmentator)

Script: `segmentation.py`

### Processar um unico arquivo

```bash
python3 segmentation.py imagens_ct/0801_clean_fdk_256.nii.gz \
  --output resultados/0801_clean_fdk_256.nii \
  --device auto
```

Saidas:

- `resultados/0801_clean_fdk_256.nii`

### Processar todos os .nii.gz em imagens_ct/

```bash
python3 segmentation.py --device auto
```

Saidas:

- `resultados/<sample>.nii`

## 3) NPY -> MHD/RAW

Script: `npy2mhd.py`

```bash
python3 npy2mhd.py 0801_clean_fdk_256.npy \
  --output data/0801_clean_fdk_256.mhd \
  --spacing 1.5 1.5 1.5 \
  --center
```

Saidas:

- `data/0801_clean_fdk_256.mhd`
- `data/0801_clean_fdk_256.raw`

## 4) Simulacao Gate (OpenGATE)

Script: `minDetec_DT_Wilson.py`

```bash
python3 minDetec_DT_Wilson.py 0 \
  --image data/0801_clean_fdk_256.mhd \
  --output-dir output \
  --tag 0801_clean_fdk_256
```

Saidas (na pasta `output/`):

- `Slide-0801_clean_fdk_256.mhd`
- `stat-0801_clean_fdk_256.txt`
- `FDK_label.mhd`

Observacoes:

- `--materials-db` e `--hu2mat` tem defaults apontando para `data/`.
- `inicio` define o angulo inicial (inteiro).

## 5) Somar resultados (dose/edep)

Script: `somarResultados.py`

```bash
python3 somarResultados.py
```

Saidas (na pasta `output/`):

- `Dose-total.mhd`
- `Edep-total.mhd`

## 6) Calcular doses e DVH

Script: `calcular_doses.py`

```bash
python3 calcular_doses.py
```

Entradas esperadas (hardcoded no topo do arquivo):

- `0801_clean_fdk_256.nii` (segmentacao)
- `output/Dose-total.mhd` (dose total)

Saidas:

- `resultados_dvh/estatisticas_dose_por_orgao.csv`
- `resultados_dvh/dvh_todos_orgaos_estilo_classico.png`
- `resultados_dvh/dvh_todos_orgaos.csv`
- `resultados_dvh/mascaras_binarias/mascara_<orgao>.nii`
- `resultados_dvh/dvh_<orgao>.png`
- `resultados_dvh/dvh_<orgao>.csv`

Se quiser mudar os arquivos de entrada, edite as variaveis:
`SEGMENTATION_PATH` e `DOSE_PATH` no topo do script.

## Fluxo completo (exemplo)

```bash
# 1) NPY -> NIfTI
python3 npy2nii.py 0801_clean_fdk_256.npy --output imagens_ct/0801_clean_fdk_256.nii.gz --spacing 1.5 1.5 1.5 --center

# 2) Segmentacao
python3 segmentation.py imagens_ct/0801_clean_fdk_256.nii.gz --output resultados/0801_clean_fdk_256.nii --device auto

# 3) NPY -> MHD/RAW
python3 npy2mhd.py 0801_clean_fdk_256.npy --output data/0801_clean_fdk_256.mhd --spacing 1.5 1.5 1.5 --center

# 4) Simulacao Gate
python3 minDetec_DT_Wilson.py 0 --image data/0801_clean_fdk_256.mhd --output-dir output --tag 0801_clean_fdk_256

# 5) Soma de dose/edep
python3 somarResultados.py

# 6) DVH e estatisticas
python3 calcular_doses.py
```
