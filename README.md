# Classificação de Flores Íris com k-NN e PCA

**Dupla:**
- Gustavo Gabriel Ripka — GRR20203935
- Arthur Barreto Godoi — GRR20224377

## Visão geral

Pipeline completo de análise do dataset Íris (UCI, 150 amostras, 4 medidas em
cm, 3 categorias), executado por um único script (`src/analise.py`, semente
fixa 42):

1. **Estatística descritiva** — média, desvio padrão e moda das 4 variáveis no
   conjunto geral e por categoria; frequência absoluta e relativa das classes
   (`results/estatisticas.txt`).
2. **Correlação comprimento × largura** da sépala e da pétala por categoria:
   dispersão com reta de regressão linear e coeficiente r de Pearson
   (`figs/sepalas_dispersao.png`, `figs/petalas_dispersao.png` e os
   respectivos `.txt`).
3. **Matriz de dispersão** das 4 dimensões com histogramas por categoria na
   diagonal (`figs/matriz_dispersao.png`).
4. **PCA** — padronização z-score, variância explicada/acumulada, loadings e
   projeção PC1×PC2 (`figs/pca.png`, `results/pca.txt`). PC1+PC2 ≈ 95,8%.
5. **Split 80/20 estratificado** (`train_test_split`, `random_state=42`).
6. **k-NN** — pipeline `StandardScaler + KNeighborsClassifier`; hiperparâmetros
   k ∈ {1, 3, ..., 21} (ímpares) e métrica de distância (euclidean/manhattan)
   escolhidos por **GridSearchCV com validação cruzada 5-fold** no treino
   (`figs/knn_cv.png`, `results/knn_gridsearch.txt`).
7. **Avaliação no teste** — acurácia, relatório de classificação e matriz de
   confusão do melhor modelo; comparação com k=1 e com o k-NN sem
   padronização (`figs/matriz_confusao.png`, `results/knn_teste.txt`).
8. Todos os números principais consolidados em `results/resultados.json`.

## Como rodar

```bash
python3 src/analise.py
```

Dependências (Python 3): `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`scipy`.

## Estrutura de pastas

```
iris-knn/
├── data/        # iris.data (dataset UCI) e iris.names (documentação)
├── src/         # analise.py — script único da análise
├── figs/        # figuras geradas (.png, 150 dpi)
└── results/     # tabelas/números (.txt) e resultados.json
```
