# 4.x Avaliacao de desempenho e escalabilidade

## Metodologia (leia antes de colar no TCC)

Todas as fontes de dados sao REAIS e medidas:

1. **Benchmarks do Snakemake (maquina local)** — diretiva `benchmark`, um `.tsv`
   por regra (wall-clock, tempo de CPU, pico de RAM). 6 etapas. -> Tabela 1, Figuras 1 e 2.
2. **Estatistica do GATE no HPC** (`stat-0.txt`): 19.996.125 primarias / 4 projecoes,
   1 thread, 2,62 h, `pps = 2119` (qualidade definitiva: 5x10^6 primarias/projecao).
3. **Estatistica do GATE local** (`stat-0801...txt`): 998.864 primarias (1E6),
   1 projecao, 1 thread, 3,28 min, `pps = 5075`.

Hardware:
- **Local:** Intel Core i5-1335U (10 nucleos / 12 threads), notebook.
- **HPC:** 2x Intel Xeon Gold 5318Y (48 nucleos fisicos / 96 logicos), 503 GB RAM.

O unico valor extrapolado e o tempo da aquisicao completa de 360 graus em cada
configuracao, obtido por `tempo = total_de_primarias / (pps_por_thread x n_threads)`,
valido porque o tempo de Monte Carlo escala linearmente com o numero de historias.
Esses pontos estao rotulados como (medido) ou (pico teorico) na Figura 4.

> Coerencia dos numeros no texto do TCC: HPC usou **5x10^6** primarias/projecao
> (`stat-0.txt`: 19.996.125 / 4); a maquina local usou **1x10^6** (`stat` local:
> 998.864). Cite esses dois valores explicitamente — eles batem com os artefatos.

---

## Texto sugerido

A avaliacao de desempenho apoiou-se nas metricas registradas automaticamente pela
diretiva `benchmark` do Snakemake. A Tabela 1 consolida os valores obtidos para a
amostra `0801_clean_fdk_256` na maquina local.

**Tabela 1 — Desempenho por etapa (maquina local, i5-1335U).**

| Etapa | Wall-clock (s) | CPU (s) | RAM pico (MB) | % do tempo total |
|---|---:|---:|---:|---:|
| Simulacao Monte Carlo (GATE, 1E6 prim.) | 211,15 | 194,05 | 2.417,8 | 81,4 |
| Segmentacao (TotalSegmentator) | 38,39 | 13,90 | 5.179,6 | 14,8 |
| Calculo de doses (DVH) | 7,64 | 7,42 | 537,2 | 2,9 |
| Conversao NPY->NIfTI | 1,23 | 0,90 | 241,5 | 0,5 |
| Conversao NPY->MHD | 0,64 | 0,44 | 281,9 | 0,2 |
| Consolidacao (soma de doses) | 0,44 | 0,00 | 3,1 | 0,2 |
| **Total** | **259,48** | — | — | **100,0** |

A Figura 1 evidencia que a simulacao Monte Carlo concentra **81,4%** do tempo total
do pipeline; as cinco etapas restantes somadas respondem por apenas 48,3 s (18,6%).
A simulacao e, portanto, o unico gargalo computacional relevante, e o desempenho
global do fluxo de trabalho e governado quase integralmente por essa etapa.

Sob a otica de memoria (Figura 2), o pico ocorre na segmentacao com o
TotalSegmentator (5,06 GB), coerente com a inferencia da rede nnU-Net, ainda assim
muito inferior aos 503 GB do servidor e compativel com estacoes modestas. O consumo
de RAM e determinado pelo tamanho do dado (volume 256^3) e pelo modelo, sendo
independente da plataforma; os valores da Tabela 1 valem tambem como referencia
para o ambiente HPC.

### Desempenho por nucleo e escalabilidade

Um resultado relevante surge da comparacao do throughput da simulacao entre as duas
plataformas. Por thread, a maquina local (i5-1335U) atinge **5.075 primarias/s**,
contra **2.119 primarias/s** do servidor (Xeon Gold 5318Y) — ou seja, o nucleo do
notebook e cerca de **2,4x mais rapido** em execucao single-thread, reflexo de sua
maior frequencia de operacao e microarquitetura mais recente, frente aos 2,1 GHz de
base do processador de servidor.

A vantagem do ambiente HPC, portanto, **nao reside na velocidade por nucleo, e sim
na densidade de paralelismo**. A aquisicao CBCT de 360 graus possui paralelismo
trivial: e decomposta em 90 invocacoes angulares independentes (`inicio` de 0 a 356,
passo 4), sem comunicacao entre processos. Cada invocacao processa 4 projecoes e foi
medida no HPC em **2,62 h** a uma thread. A Figura 3 mostra o tempo total em funcao do
numero de processos concorrentes, `T(p) = ceil(90/p) x 2,62 h`: executada serialmente
exigiria ~236 h (~9,8 dias); distribuida pelos 48 nucleos fisicos cai para **~5,2 h**;
e com 90-96 processos (aproveitando o hyperthreading) aproxima-se de **~2,6 h**,
limite imposto pela duracao de um unico job. O comportamento e praticamente linear,
como esperado para uma carga sem dependencias.

A Figura 4 sintetiza o impacto pratico para a aquisicao completa na qualidade
definitiva (5x10^6 primarias/projecao, 1,8x10^9 primarias no total). Embora o notebook
seja mais rapido por nucleo (98,5 h serial, contra 236 h do servidor), seus 10 nucleos
em um chip de baixo consumo nao sustentam carga total prolongada; mesmo no melhor caso
teorico (12 threads, ~8,2 h, desconsiderando throttling termico) nao alcanca o servidor.
O HPC, ao distribuir os 90 jobs por seus 48-96 nucleos com clock sustentado, conclui a
aquisicao em **~2,6 h**, viabilizando ainda **cinco vezes mais estatistica** por
projecao (5x10^6 contra 1x10^6) dentro de um tempo pratico. Esse ganho de throughput
agregado e de robustez operacional — e nao a velocidade por nucleo — e o que justifica
metodologicamente a producao dos resultados definitivos no ambiente de alto desempenho,
ao passo que a validacao funcional integral do pipeline foi conduzida na maquina local.
