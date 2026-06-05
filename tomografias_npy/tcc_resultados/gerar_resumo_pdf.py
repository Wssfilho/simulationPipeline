#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera um PDF com o resumo passo a passo da analise de desempenho do TCC."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, HRFlowable, KeepTogether)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\wilso\Downloads\resumo_tcc_desempenho.pdf"

NAVY = colors.HexColor("#1f3b63")
RED = colors.HexColor("#c0392b")
GRAY = colors.HexColor("#555555")
LGRAY = colors.HexColor("#eef1f5")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=15,
                    spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=RED, fontSize=11.5,
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=9.7, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=5)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=12, bulletIndent=2,
                        spaceAfter=2)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], textColor=NAVY, fontSize=20,
                       leading=24)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10.5, textColor=GRAY,
                     alignment=TA_CENTER, spaceAfter=2)
CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontSize=8, textColor=GRAY,
                     alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)

story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def p(t): story.append(Paragraph(t, BODY))
def b(t): story.append(Paragraph("&bull;&nbsp; " + t, BULLET))
def sp(h=6): story.append(Spacer(1, h))


def fig(name, caption, width=15*cm):
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
        img = Image(path, width=width, height=width*ih/iw)
        story.append(KeepTogether([img, Paragraph(caption, CAP)]))


# ---------------------------------------------------------------- capa
story.append(Spacer(1, 10))
story.append(Paragraph("Analise de Desempenho e Paralelizacao", TITLE))
story.append(Paragraph("Pipeline de simulacao de dose CBCT (GATE 10 + Snakemake)", SUB))
story.append(Paragraph("Resumo passo a passo &mdash; TCC", SUB))
sp(4)
story.append(HRFlowable(width="100%", color=NAVY, thickness=1.2))
sp(10)

# ---------------------------------------------------------------- 1
h1("1. Objetivo da analise")
p("Avaliar e documentar o desempenho do pipeline e justificar, de forma honesta, "
  "por que a producao dos resultados definitivos foi feita no servidor de alto "
  "desempenho (HPC) e nao na maquina local.")
p("<b>Pergunta inicial:</b> &ldquo;da para estimar os dados do HPC tendo so os "
  "locais + a configuracao?&rdquo; <b>Resposta:</b> em parte. O que importa para o "
  "desempenho (a simulacao) foi <b>medido</b> de verdade; as demais etapas sao "
  "desprezveis e nao precisam ser fabricadas.")

# ---------------------------------------------------------------- 2
h1("2. Hardware comparado")
data = [["", "Servidor HPC", "Notebook (local)"],
        ["Processador", "2x Xeon Gold 5318Y @2,1 GHz", "Intel Core i5-1335U"],
        ["Nucleos fisicos", "48 (24 x 2 sockets)", "10 (2 performance + 8 efic.)"],
        ["Threads logicos", "96", "12"],
        ["RAM", "503 GB", "7,6 GB (ambiente WSL)"],
        ["GPUs", "L40S 46GB, A5000, A4000", "(integrada)"]]
t = Table(data, colWidths=[3.4*cm, 6.2*cm, 6.2*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGRAY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
story.append(t)
sp(4)
p("Descoberta-chave: o nucleo do <b>notebook e ~2,4x mais rapido</b> que o do Xeon "
  "(5075 vs 2119 particulas/s por thread). A vantagem do HPC <b>nao</b> e velocidade "
  "por nucleo &mdash; e <b>quantidade</b> de nucleos.")

# ---------------------------------------------------------------- 3
h1("3. Onde o tempo do pipeline e gasto")
p("As 6 etapas foram medidas pela diretiva <font face='Courier'>benchmark</font> do "
  "Snakemake. A <b>simulacao Monte Carlo (GATE) concentra 81,4%</b> do tempo total; "
  "as outras cinco etapas somadas dao apenas 48,3 s (18,6%).")
p("<b>Conclusao:</b> a simulacao e o unico gargalo relevante. Por isso o estudo de "
  "desempenho foca nela &mdash; e as demais etapas, por serem desprezveis, nao "
  "precisaram de dados fabricados do HPC.")
fig("fig1_perfil_etapas.png", "Figura 1 - Tempo por etapa (maquina local).")

# ---------------------------------------------------------------- 4
h1("4. Memoria (RAM)")
p("O pico de memoria ocorre na segmentacao com o TotalSegmentator: <b>5,06 GB</b>. "
  "Cabe folgado nas duas maquinas. Como o consumo depende do tamanho do dado "
  "(volume 256^3) e do modelo, ele e <b>independente da plataforma</b> &mdash; os "
  "valores locais valem como referencia tambem para o HPC.")

# ---------------------------------------------------------------- 5
h1("5. A simulacao e o paralelismo")
b("A aquisicao de <b>360 graus</b> = 360 projecoes ao redor do paciente.")
b("<b>n = 4</b>: cada execucao do script faz 4 projecoes. Logo, <b>360 / 4 = 90 "
  "execucoes</b> cobrem a volta completa.")
b("As 90 execucoes sao <b>independentes</b> (nao trocam informacao) &mdash; "
  "&ldquo;paralelismo trivial&rdquo;: podem rodar todas ao mesmo tempo.")
b("<b>Leva</b> = uma rodada de execucoes rodando simultaneamente. O numero de "
  "levas determina o tempo total.")

# ---------------------------------------------------------------- 6
h1("6. A formula da escalabilidade")
fb = ParagraphStyle("fb", parent=BODY, alignment=TA_CENTER, fontSize=11,
                    textColor=NAVY, spaceBefore=4, spaceAfter=4)
story.append(Paragraph("<b>Tempo total = teto(90 / nucleos) &times; tempo de uma execucao</b>", fb))
p("Quanto mais nucleos, mais execucoes cabem por leva, menos levas, menos tempo. "
  "<font face='Courier'>teto()</font> arredonda para cima, porque uma execucao e "
  "indivisivel (nao da para fazer meia leva).")
fig("fig3_escalabilidade.png", "Figura 2 - Tempo cai conforme aumentam os nucleos (HPC).", 14*cm)

# ---------------------------------------------------------------- 7
h1("7. Tempo da aquisicao completa em cada maquina")
data2 = [["Configuracao", "1 execucao", "Por leva", "Levas", "Total"],
         ["Servidor HPC (96 threads)", "2,62 h", "90", "teto(90/96)=1", "2,6 h"],
         ["Servidor HPC (48 nucleos)", "2,62 h", "48", "teto(90/48)=2", "5,2 h"],
         ["Notebook - melhor caso", "59 min", "12", "teto(90/12)=8", "7,9 h"],
         ["Notebook - REAL (limite RAM)", "59 min", "2", "teto(90/2)=45", "44 h"]]
t2 = Table(data2, colWidths=[5.6*cm, 2.4*cm, 2.0*cm, 3.0*cm, 2.2*cm])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGRAY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TEXTCOLOR", (4, 1), (4, 2), RED),
    ("FONTNAME", (4, 1), (4, -1), "Helvetica-Bold"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
story.append(t2)
sp(4)
b("<b>Servidor:</b> limitado pelos <b>nucleos</b> (RAM de 503 GB sobra). As 90 "
  "execucoes cabem em 1 leva -&gt; 2,6 h.")
b("<b>Notebook:</b> limitado pela <b>RAM</b> (7,6 GB / 2,8 GB por execucao = so 2 "
  "por vez) -&gt; 45 levas -&gt; ~44 h reais. Os 7,9 h sao so o melhor caso "
  "teorico (se tivesse RAM para 12 execucoes).")
fig("fig4_comparacao_maquinas.png", "Figura 3 - Tempo total por maquina (mesma qualidade, 5x10^6).", 14*cm)

# ---------------------------------------------------------------- 8
h1("8. Conceitos-chave (glossario)")
b("<b>Execucao (ou &lsquo;job&rsquo;):</b> uma rodada do script de simulacao = 4 "
  "projecoes (n=4). Sao 90 para cobrir os 360 graus.")
b("<b>Leva:</b> rodada de execucoes que rodam ao mesmo tempo. teto(90/nucleos) = "
  "numero de levas.")
b("<b>Nucleo fisico vs thread logico:</b> o HPC tem 48 nucleos fisicos que, com "
  "hyperthreading, viram 96 threads. Mas 2 threads dividem 1 nucleo &mdash; nao e o "
  "dobro de poder (ganho de ~20-40%). Por isso o tempo real fica entre 2,6 h "
  "(96 threads) e 5,2 h (48 nucleos).")
b("<b>n = 4:</b> tamanho do &lsquo;pedaco&rsquo; (projecoes por execucao). Controla "
  "em quantas execucoes os 360 graus sao divididos.")
b("<b>pps (particulas/segundo):</b> velocidade da simulacao por thread. Mede o "
  "desempenho por nucleo, independente de quantas projecoes.")

# ---------------------------------------------------------------- 9
h1("9. Integracao no Snakemake")
p("<b>Antes:</b> um script de shell externo (<font face='Courier'>for ... nohup &amp;</font>) "
  "disparava as 90 execucoes &mdash; rapido, mas sem controle (saturava maquina fraca).")
p("<b>Agora:</b> a varredura angular foi integrada ao Snakemake por um "
  "<font face='Courier'>wildcard</font> de angulo. A propria ferramenta gera as 90 "
  "execucoes e paraleliza com <font face='Courier'>--cores N</font> (N execucoes por "
  "vez; o resto e enfileirado). Nao precisa de 90 nucleos.")
b("No HPC: <font face='Courier'>snakemake --cores 48</font>")
b("No notebook (demo): <font face='Courier'>snakemake --cores 2 --latency-wait 30 "
  "--config gate_angulo_final=16</font>")

# ---------------------------------------------------------------- 10
h1("10. Problemas encontrados e resolvidos")
b("<b>Corrida no FDK_label:</b> o arquivo de rotulos era gravado sem identificador, "
  "entao varias execucoes paralelas escreviam por cima umas das outras. "
  "<b>Correcao:</b> gravar so na primeira execucao (inicio==0) -&gt; um unico "
  "FDK_label.mhd/raw/json.")
b("<b>OOM (falta de memoria) no notebook:</b> com --cores 4, 4 execucoes x 2,8 GB "
  "estouravam os 7,6 GB e o sistema matava processos. <b>Correcao:</b> --cores 2.")
b("<b>Latencia do WSL:</b> o Snakemake nao &lsquo;via&rsquo; os arquivos a tempo. "
  "<b>Correcao:</b> --latency-wait 30.")

# ---------------------------------------------------------------- 11
h1("11. Integridade dos dados (para a banca)")
b("<b>Medido de verdade:</b> tempos e RAM por etapa; pps das duas maquinas; tempo "
  "de uma execucao (59 min local, 2,62 h HPC).")
b("<b>Estimado (extrapolacao linear, rotulado):</b> os tempos das aquisicoes "
  "completas (7,9 h; 44 h; 2,6 h; 5,2 h) = tempo de uma execucao x numero de levas. "
  "Valido porque o tempo de Monte Carlo escala linearmente com o numero de particulas.")
b("<b>Coerencia obrigatoria no texto:</b> o HPC usou <b>5x10^6</b> primarias por "
  "projecao; o notebook usou <b>1x10^6</b> nos benchmarks. A comparacao de tempo "
  "(fig. 3) usa <b>5x10^6 nas duas</b> &mdash; mesma qualidade, comparacao justa.")
b("<b>Nao foi inventado nada:</b> os tempos das demais ferramentas no HPC nao foram "
  "fabricados &mdash; por serem desprezveis, sua variacao nao muda a conclusao.")

# ---------------------------------------------------------------- 12
h1("12. Entregaveis gerados")
b("Figuras (300 dpi, escala linear): perfil por etapa, memoria, escalabilidade, "
  "comparacao entre maquinas.")
b("<font face='Courier'>desempenho.tex</font>: secao de resultados pronta (tabelas, "
  "equacoes, o calculo dos tempos passo a passo).")
b("<font face='Courier'>paralelizacao.tex</font>: paragrafo + apendice com o script.")
b("Snakefile modificado (wildcard de angulo) e correcoes no minDetec_DT.py, em "
  "branch git separada (<font face='Courier'>feature/snakemake-360-angulos</font>).")

sp(8)
story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=0.6))
story.append(Paragraph("Documento gerado automaticamente como resumo da analise. "
                       "Numeros conferidos contra os arquivos stat-*.txt e os "
                       "benchmarks do Snakemake.", CAP))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=1.6*cm, bottomMargin=1.6*cm,
                        title="Resumo - Desempenho do pipeline (TCC)")
doc.build(story)
print("PDF gerado:", OUT)
