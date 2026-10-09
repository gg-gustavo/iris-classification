# -*- coding: utf-8 -*-
"""
Análise do dataset Íris (UCI) com estatística descritiva, correlações,
PCA e classificação com k-NN (pipeline StandardScaler + KNeighborsClassifier).

Trabalho T4 — Classificação de Flores Íris com k-NN e PCA
Dupla: Gustavo Gabriel Ripka (GRR20203935) e Arthur Barreto Godoi (GRR20224377)

Uso:
    python3 src/analise.py

Gera:
    figs/sepalas_dispersao.png, figs/petalas_dispersao.png,
    figs/matriz_dispersao.png, figs/pca.png, figs/knn_cv.png,
    figs/matriz_confusao.png
    results/estatisticas.txt, results/sepalas_correlacao.txt,
    results/petalas_correlacao.txt, results/pca.txt, results/split.txt,
    results/knn_gridsearch.txt, results/knn_teste.txt, results/resultados.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica (salva arquivos)
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------
# Configurações globais
# ----------------------------------------------------------------------
SEED = 42                      # semente global fixa (reprodutibilidade)
np.random.seed(SEED)

# Caminhos (relativos à raiz do projeto: iris-knn/)
RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "iris.data"
FIGS = RAIZ / "figs"
RESULTS = RAIZ / "results"

# Colunas do CSV (sem cabeçalho) e nomes legíveis em português
COLUNAS = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
ALVO = "classe"
NOMES = {
    "sepal_length_cm": "Comprimento da sépala (cm)",
    "sepal_width_cm": "Largura da sépala (cm)",
    "petal_length_cm": "Comprimento da pétala (cm)",
    "petal_width_cm": "Largura da pétala (cm)",
}
ORDEM_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
NOMES_CURTOS = {"Iris-setosa": "setosa", "Iris-versicolor": "versicolor",
                "Iris-virginica": "virginica"}
CORES = {"Iris-setosa": "tab:blue", "Iris-versicolor": "tab:orange",
         "Iris-virginica": "tab:green"}

# Estilo dos gráficos
for estilo in ("seaborn-v0_8", "seaborn", "ggplot"):
    try:
        plt.style.use(estilo)
        break
    except OSError:
        continue
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 12, "axes.labelsize": 11})


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def carregar_dados() -> pd.DataFrame:
    """Lê data/iris.data (CSV sem cabeçalho) e devolve um DataFrame."""
    df = pd.read_csv(DADOS, header=None, names=COLUNAS + [ALVO])
    df = df.dropna()                       # remove eventual linha em branco
    df[ALVO] = df[ALVO].str.strip()        # remove espaços extras no rótulo
    return df


def moda_formatada(serie: pd.Series) -> str:
    """Moda da série; se houver empate, junta os valores com ' / '."""
    modas = serie.mode()
    return " / ".join(f"{v:g}" for v in modas)


def salvar_texto(nome: str, conteudo: str) -> None:
    """Salva uma string em results/<nome>."""
    (RESULTS / nome).write_text(conteudo, encoding="utf-8")


def salvar_figura(fig: plt.Figure, nome: str) -> None:
    """Aplica tight_layout e salva a figura em figs/<nome>."""
    fig.tight_layout()
    fig.savefig(FIGS / nome, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 1) Estatística descritiva
# ----------------------------------------------------------------------
def analise_estatistica(df: pd.DataFrame) -> dict:
    """Média, desvio padrão e moda (geral e por classe) + frequências.

    Salva results/estatisticas.txt e devolve estrutura para o JSON.
    """
    linhas = ["=" * 78,
              "ESTATÍSTICA DESCRITIVA — DATASET ÍRIS (150 amostras)",
              "=" * 78, ""]

    # --- (a) conjunto geral ---
    geral = df[COLUNAS]
    tab_geral = pd.DataFrame({
        "média": geral.mean(),
        "desvio_padrão": geral.std(),
        "mínimo": geral.min(),
        "máximo": geral.max(),
        "moda": [moda_formatada(geral[c]) for c in COLUNAS],
    })
    linhas += ["--- (a) Conjunto geral (todas as 150 flores) ---",
               tab_geral.to_string(float_format=lambda v: f"{v:.3f}"), ""]

    # --- (b) por categoria ---
    linhas.append("--- (b) Por categoria ---")
    resumo_por_classe = {}
    for classe in ORDEM_CLASSES:
        sub = df.loc[df[ALVO] == classe, COLUNAS]
        tab = pd.DataFrame({
            "média": sub.mean(),
            "desvio_padrão": sub.std(),
            "mínimo": sub.min(),
            "máximo": sub.max(),
            "moda": [moda_formatada(sub[c]) for c in COLUNAS],
        })
        linhas += [f"\n>> {classe} (n = {len(sub)})",
                   tab.to_string(float_format=lambda v: f"{v:.3f}")]
        resumo_por_classe[classe] = {
            c: {"media": float(sub[c].mean()), "desvio_padrao": float(sub[c].std()),
                "moda": moda_formatada(sub[c])} for c in COLUNAS}

    # --- frequências das categorias ---
    freq_abs = df[ALVO].value_counts().reindex(ORDEM_CLASSES)
    freq_rel = (freq_abs / len(df) * 100).round(2)
    linhas += ["", "--- Frequência das categorias ---",
               pd.DataFrame({"absoluta": freq_abs, "relativa_%": freq_rel}).to_string()]

    salvar_texto("estatisticas.txt", "\n".join(linhas) + "\n")

    # estrutura para o JSON
    return {
        "geral": {c: {"media": float(geral[c].mean()),
                      "desvio_padrao": float(geral[c].std()),
                      "moda": moda_formatada(geral[c])} for c in COLUNAS},
        "por_classe": resumo_por_classe,
        "frequencias": {"absoluta": {k: int(v) for k, v in freq_abs.items()},
                        "relativa_%": {k: float(v) for k, v in freq_rel.items()}},
    }


# ----------------------------------------------------------------------
# 2) e 3) Dispersão comprimento × largura, com regressão e Pearson r
# ----------------------------------------------------------------------
def dispersao_com_regressao(df: pd.DataFrame, parte: str) -> dict:
    """Scatter comprimento × largura por categoria + reta de regressão + r.

    'parte' deve ser 'sepal' ou 'petal'. Gera figs/<parte>s_dispersao.png e
    results/<parte>s_correlacao.txt. Devolve dicionário para o JSON.
    """
    x_col, y_col = f"{parte}_length_cm", f"{parte}_width_cm"
    titulo = "sépala" if parte == "sepal" else "pétala"
    prefixo = "sepalas" if parte == "sepal" else "petalas"

    fig, ax = plt.subplots(figsize=(8, 6))
    linhas_txt = [f"CORRELAÇÃO COMPRIMENTO × LARGURA DA {titulo.upper()}",
                  "(regressão linear simples por categoria)", "-" * 60]
    resultado = {}

    for classe in ORDEM_CLASSES:
        sub = df[df[ALVO] == classe]
        x, y = sub[x_col].values, sub[y_col].values

        # dispersão
        ax.scatter(x, y, s=45, alpha=0.75, color=CORES[classe],
                   label=NOMES_CURTOS[classe], edgecolor="white", linewidth=0.5)

        # regressão linear (mínimos quadrados) + correlação de Pearson
        reg = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, reg.intercept + reg.slope * xs, color=CORES[classe],
                linewidth=2, linestyle="--",
                label=f"{NOMES_CURTOS[classe]}: ajuste linear")

        sinal = "+" if reg.intercept >= 0 else "-"
        linhas_txt += [
            f"\n{classe}:",
            f"  r de Pearson      = {reg.rvalue:+.4f}",
            f"  p-valor           = {reg.pvalue:.4g}",
            f"  equação da reta   : largura = {reg.slope:.4f}·comprimento {sinal} {abs(reg.intercept):.4f}",
        ]
        resultado[classe] = {"r_pearson": float(reg.rvalue),
                             "p_valor": float(reg.pvalue),
                             "inclinação": float(reg.slope),
                             "intercepto": float(reg.intercept)}

    ax.set_xlabel(NOMES[x_col])
    ax.set_ylabel(NOMES[y_col])
    ax.set_title(f"Comprimento × largura da {titulo} por categoria\n"
                 "(linhas tracejadas: regressão linear por categoria)")
    ax.legend(title="Categoria", loc="best")
    salvar_figura(fig, f"{prefixo}_dispersao.png")
    salvar_texto(f"{prefixo}_correlacao.txt", "\n".join(linhas_txt) + "\n")
    return resultado


# ----------------------------------------------------------------------
# 4) Matriz de dispersão (scatter matrix manual) com histogramas
# ----------------------------------------------------------------------
def matriz_dispersao(df: pd.DataFrame) -> None:
    """Matriz 4×4: histogramas por categoria na diagonal, dispersões fora."""
    n = len(COLUNAS)
    fig, axes = plt.subplots(n, n, figsize=(13, 11))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:  # diagonal: histogramas sobrepostos por categoria
                for classe in ORDEM_CLASSES:
                    vals = df.loc[df[ALVO] == classe, COLUNAS[i]]
                    ax.hist(vals, bins=15, alpha=0.6, color=CORES[classe],
                            label=NOMES_CURTOS[classe])
                if i == 0:
                    ax.legend(fontsize=8, title="categoria", title_fontsize=8)
            else:       # fora da diagonal: dispersão colorida por categoria
                for classe in ORDEM_CLASSES:
                    sub = df[df[ALVO] == classe]
                    ax.scatter(sub[COLUNAS[j]], sub[COLUNAS[i]], s=12,
                               alpha=0.7, color=CORES[classe])
            if i == n - 1:
                ax.set_xlabel(NOMES[COLUNAS[j]], fontsize=9)
            if j == 0:
                ax.set_ylabel(NOMES[COLUNAS[i]], fontsize=9)

    fig.suptitle("Distribuição das categorias nas 4 dimensões — "
                 "matriz de dispersão (diagonal: histogramas)", fontsize=13)
    salvar_figura(fig, "matriz_dispersao.png")


# ----------------------------------------------------------------------
# 5) PCA
# ----------------------------------------------------------------------
def analise_pca(df: pd.DataFrame) -> dict:
    """PCA com padronização z-score: scree + projeção PC1×PC2.

    Gera figs/pca.png e results/pca.txt; devolve números para o JSON.
    """
    X = df[COLUNAS].values

    # padronização z-score (média 0, desvio 1) antes do PCA
    X_pad = StandardScaler().fit_transform(X)

    pca = PCA(random_state=SEED).fit(X_pad)
    var_exp = pca.explained_variance_ratio_            # variância explicada
    var_acum = np.cumsum(var_exp)                      # acumulada
    loadings = pca.components_                         # pesos (autovetores)
    X_proj = pca.transform(X_pad)                      # projeções (scores)

    # ---- arquivo de texto ----
    linhas = ["ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) — dados padronizados (z-score)",
              "-" * 70, "",
              "Variância explicada por componente:"]
    for i, (v, a) in enumerate(zip(var_exp, var_acum), start=1):
        linhas.append(f"  PC{i}: {v*100:6.2f}%   (acumulada: {a*100:6.2f}%)")
    linhas += ["", "Loadings (componentes_) — peso de cada variável em cada PC:",
               f"  {'variável':<20s}" + "".join(f"{f'PC{i+1}':>9s}" for i in range(4))]
    for k, col in enumerate(COLUNAS):
        linhas.append(f"  {col:<20s}" + "".join(f"{loadings[i, k]:>9.3f}" for i in range(4)))
    linhas += ["\nObservação: PC1 e PC2 concentram "
               f"{var_acum[1]*100:.2f}% da variância total."]
    salvar_texto("pca.txt", "\n".join(linhas) + "\n")

    # ---- figura com 2 subplots ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # subplot 1: scree (barras) + variância acumulada (linha)
    comps = np.arange(1, len(var_exp) + 1)
    ax1.bar(comps, var_exp * 100, color="tab:blue", alpha=0.8,
            label="Variância explicada")
    ax1.plot(comps, var_acum * 100, "o-", color="tab:red", label="Acumulada")
    for x, v, a in zip(comps, var_exp, var_acum):
        ax1.annotate(f"{v*100:.1f}%", (x, v * 100), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8)
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Variância explicada (%)")
    ax1.set_title("Variância explicada por componente (scree)")
    ax1.set_xticks(comps)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="center right")
    ax1.grid(alpha=0.3)

    # subplot 2: projeção nos 2 primeiros PCs
    for classe in ORDEM_CLASSES:
        m = (df[ALVO] == classe).values
        ax2.scatter(X_proj[m, 0], X_proj[m, 1], s=40, alpha=0.8,
                    color=CORES[classe], label=NOMES_CURTOS[classe],
                    edgecolor="white", linewidth=0.5)
    ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% da variância)")
    ax2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% da variância)")
    ax2.set_title("Projeção das amostras em PC1 × PC2")
    ax2.legend(title="categoria")
    ax2.grid(alpha=0.3)

    salvar_figura(fig, "pca.png")

    return {"variancia_explicada_%": [round(v * 100, 3) for v in var_exp],
            "variancia_acumulada_%": [round(a * 100, 3) for a in var_acum],
            "variancia_explicada_pc1_pc2_%": round(var_acum[1] * 100, 3),
            "loadings": [[round(float(v), 4) for v in linha] for linha in loadings]}


# ----------------------------------------------------------------------
# 6) Split 80/20 estratificado
# ----------------------------------------------------------------------
def dividir_treino_teste(df: pd.DataFrame) -> tuple:
    """Split 80/20 estratificado (SEED fixa); registra results/split.txt."""
    X = df[COLUNAS].values
    y = df[ALVO].values
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cont_treino = pd.Series(y_treino).value_counts().reindex(ORDEM_CLASSES)
    cont_teste = pd.Series(y_teste).value_counts().reindex(ORDEM_CLASSES)
    linhas = ["DIVISÃO TREINO/TESTE (80/20, estratificada, random_state=42)",
              "-" * 60,
              f"Treino: {len(y_treino)} amostras",
              f"Teste : {len(y_teste)} amostras", "",
              pd.DataFrame({"treino": cont_treino, "teste": cont_teste}).to_string(),
              "", "Proporções mantidas por classe (estratificação): "
              "80% treino / 20% teste em cada categoria."]
    salvar_texto("split.txt", "\n".join(linhas) + "\n")

    return X_treino, X_teste, y_treino, y_teste, {"treino": int(len(y_treino)),
                                                  "teste": int(len(y_teste)),
                                                  "composicao_treino": {k: int(v) for k, v in cont_treino.items()},
                                                  "composicao_teste": {k: int(v) for k, v in cont_teste.items()}}


# ----------------------------------------------------------------------
# 7) k-NN: seleção de hiperparâmetros com validação cruzada (5-fold)
# ----------------------------------------------------------------------
def grid_search_knn(X_treino, y_treino) -> tuple:
    """GridSearchCV (cv=5) sobre k ∈ {1,3,...,21} e métrica da distância.

    Gera figs/knn_cv.png e results/knn_gridsearch.txt.
    Devolve (grid, dicionário-resumo para o JSON).
    """
    ks = list(range(1, 22, 2))  # 1, 3, 5, ..., 21 (ímpares)

    # pipeline da aula: padronização + k-NN (o scaler é reajustado em cada fold)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("knn", KNeighborsClassifier())])

    grade = {"knn__n_neighbors": ks, "knn__metric": ["euclidean", "manhattan"]}
    grid = GridSearchCV(pipe, grade, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_treino, y_treino)

    melhor_k = grid.best_params_["knn__n_neighbors"]
    melhor_metrica = grid.best_params_["knn__metric"]
    melhor_medio = grid.best_score_
    idx = int(np.flatnonzero(
        (np.array(grid.cv_results_["param_knn__n_neighbors"]) == melhor_k) &
        (np.array(grid.cv_results_["param_knn__metric"]) == melhor_metrica))[0])
    melhor_std = grid.cv_results_["std_test_score"][idx]

    # ---- tabela completa k × métrica ----
    res = pd.DataFrame(grid.cv_results_)
    tab = (res[["param_knn__n_neighbors", "param_knn__metric",
                "mean_test_score", "std_test_score"]]
           .assign(**{
               "param_knn__n_neighbors": lambda d: d["param_knn__n_neighbors"].astype(int),
               "mean_test_score": lambda d: d["mean_test_score"].round(4),
               "std_test_score": lambda d: d["std_test_score"].round(4)})
           .sort_values(["param_knn__n_neighbors", "param_knn__metric"]))
    linhas = ["SELEÇÃO DE HIPERPARÂMETROS DO k-NN — GridSearchCV (cv=5, accuracy)",
              "Pipeline: StandardScaler + KNeighborsClassifier", "-" * 70,
              f"Melhor k     : {melhor_k}",
              f"Melhor métrica de distância: {melhor_metrica}",
              f"Acurácia média de validação cruzada: {melhor_medio:.4f} ± {melhor_std:.4f}",
              "", "Resultados completos do grid (acurácia média ± desvio em 5 folds):",
              tab.to_string(index=False)]
    salvar_texto("knn_gridsearch.txt", "\n".join(linhas) + "\n")

    # ---- figura: acurácia média vs k para as duas métricas ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metrica, cor in [("euclidean", "tab:blue"), ("manhattan", "tab:red")]:
        m = (res["param_knn__metric"] == metrica).values
        ks_m = res.loc[m, "param_knn__n_neighbors"].astype(int).values
        medias = res.loc[m, "mean_test_score"].values
        desvios = res.loc[m, "std_test_score"].values
        ordem = np.argsort(ks_m)
        ax.errorbar(ks_m[ordem], medias[ordem], yerr=desvios[ordem],
                    marker="o", capsize=3, color=cor, label=f"distância {metrica}")
    ax.axvline(melhor_k, color="gray", linestyle=":", alpha=0.8,
               label=f"melhor k = {melhor_k} ({melhor_metrica})")
    ax.set_xlabel("k (número de vizinhos)")
    ax.set_ylabel("Acurácia média (validação cruzada 5-fold)")
    ax.set_title("k-NN: acurácia média ± desvio em função de k e da métrica")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.3)
    salvar_figura(fig, "knn_cv.png")

    return grid, {
        "melhor_k": int(melhor_k),
        "melhor_metrica": melhor_metrica,
        "acuracia_cv_media": round(float(melhor_medio), 4),
        "acuracia_cv_std": round(float(melhor_std), 4),
        "ks_avaliados": ks,
    }


# ----------------------------------------------------------------------
# 8) e 9) Avaliação final no conjunto de teste
# ----------------------------------------------------------------------
def avaliar_teste(grid, X_treino, X_teste, y_treino, y_teste) -> dict:
    """Avalia o melhor modelo no teste + k=1 (NN) + versão sem padronização."""
    melhor_k = grid.best_params_["knn__n_neighbors"]
    melhor_metrica = grid.best_params_["knn__metric"]

    # melhor modelo (GridSearchCV com refit=True já retreina no treino todo)
    y_pred = grid.predict(X_teste)
    acc_melhor = float(np.mean(y_pred == y_teste))
    relatorio = classification_report(y_teste, y_pred, output_dict=True)
    cm = confusion_matrix(y_teste, y_pred, labels=ORDEM_CLASSES)

    # comparação: k=1 (vizinho mais próximo)
    nn1 = Pipeline([("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=1, metric="euclidean"))])
    nn1.fit(X_treino, y_treino)
    acc_k1 = float(nn1.score(X_teste, y_teste))

    # comparação: melhor k SEM padronização (efeito do scaler)
    knn_cru = KNeighborsClassifier(n_neighbors=melhor_k, metric=melhor_metrica)
    knn_cru.fit(X_treino, y_treino)
    acc_sem_scaler = float(knn_cru.score(X_teste, y_teste))

    # ---- arquivo de texto ----
    linhas = ["AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (30 amostras)", "-" * 60,
              f"Modelo: StandardScaler + k-NN (k={melhor_k}, métrica={melhor_metrica})",
              f"Acurácia no teste: {acc_melhor:.4f} ({acc_melhor*100:.1f}%)", "",
              "Relatório de classificação (precision/recall/f1 por classe):",
              classification_report(y_teste, y_pred, digits=3), "",
              "Matriz de confusão (linhas: classe real; colunas: classe prevista):",
              " " * 16 + "".join(f"{NOMES_CURTOS[c]:>12s}" for c in ORDEM_CLASSES)]
    for i, classe in enumerate(ORDEM_CLASSES):
        linhas.append(f"  {NOMES_CURTOS[classe]:>12s}" + "".join(f"{cm[i, j]:>12d}" for j in range(3)))
    linhas += ["", "--- COMPARAÇÕES ---",
               f"k = 1 (NN, euclidiana, com scaler)  : {acc_k1:.4f}",
               f"melhor k = {melhor_k} ({melhor_metrica}, com scaler)     : {acc_melhor:.4f}",
               f"melhor k SEM padronização (sem scaler): {acc_sem_scaler:.4f}",
               "", "Observação: no Íris as 4 variáveis têm escalas próximas (0,1–7 cm),",
               "por isso o k-NN sem padronização pode ter desempenho semelhante (ou",
               "até superior por azar amostral). A padronização permanece recomendada",
               "como boa prática: em distâncias, variáveis com escala maior dominam o",
               "cálculo quando as escalas diferem, e o scaler torna o k-NN invariante",
               "a mudanças de unidade de medida."]
    salvar_texto("knn_teste.txt", "\n".join(linhas) + "\n")

    # ---- figura: matriz de confusão ----
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), [NOMES_CURTOS[c] for c in ORDEM_CLASSES])
    ax.set_yticks(range(3), [NOMES_CURTOS[c] for c in ORDEM_CLASSES])
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"Matriz de confusão — k-NN (k={melhor_k}, {melhor_metrica})\n"
                 f"acurácia no teste = {acc_melhor:.3f}")
    for i in range(3):
        for j in range(3):
            cor = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=13, color=cor)
    fig.colorbar(im, ax=ax, label="nº de amostras")
    salvar_figura(fig, "matriz_confusao.png")

    return {"acuracia_teste_melhor_modelo": round(acc_melhor, 4),
            "acuracia_teste_k1": round(acc_k1, 4),
            "acuracia_teste_sem_scaler": round(acc_sem_scaler, 4),
            "classification_report": relatorio,
            "matriz_confusao": cm.tolist()}


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    print("Lendo dados de", DADOS)
    df = carregar_dados()
    print(f"  -> {len(df)} amostras, {len(COLUNAS)} variáveis, "
          f"{df[ALVO].nunique()} categorias")

    resultados = {}

    print("[1/7] Estatística descritiva ...")
    resultados["estatisticas"] = analise_estatistica(df)

    print("[2/7] Dispersão e correlação — sépalas ...")
    resultados["correlacao_sepalas"] = dispersao_com_regressao(df, "sepal")

    print("[3/7] Dispersão e correlação — pétalas ...")
    resultados["correlacao_petalas"] = dispersao_com_regressao(df, "petal")

    print("[4/7] Matriz de dispersão ...")
    matriz_dispersao(df)

    print("[5/7] PCA ...")
    resultados["pca"] = analise_pca(df)

    print("[6/7] Split 80/20 estratificado ...")
    X_treino, X_teste, y_treino, y_teste, resumo_split = dividir_treino_teste(df)
    resultados["split"] = resumo_split

    print("[7/7] k-NN: GridSearchCV (5-fold) + avaliação no teste ...")
    grid, resumo_grid = grid_search_knn(X_treino, y_treino)
    resultados["knn"] = {**resumo_grid,
                         **avaliar_teste(grid, X_treino, X_teste, y_treino, y_teste)}

    # 10) JSON consolidado com todos os números principais
    with open(RESULTS / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\nConcluído! Arquivos gerados em figs/ e results/.")
    print(f"  Melhor modelo: k={resultados['knn']['melhor_k']} "
          f"({resultados['knn']['melhor_metrica']}), "
          f"CV={resultados['knn']['acuracia_cv_media']:.4f}, "
          f"teste={resultados['knn']['acuracia_teste_melhor_modelo']:.4f}")


if __name__ == "__main__":
    main()
