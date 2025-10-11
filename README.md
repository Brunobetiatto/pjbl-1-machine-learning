# Framework de Classificação de Machine Learning

Este repositório contém `algoritmos_class.py`, um script Python modular e robusto projetado para simplificar e automatizar o pipeline de treinamento, validação e análise de modelos de classificação. O framework encapsula as melhores práticas de pré-processamento, avaliação e interpretação de modelos, oferecendo uma interface unificada para uma vasta gama de algoritmos do Scikit-learn, bem como implementações de ensemble personalizadas como Blending.

***

## Funcionalidades Principais

* **Interface Unificada**: Treine e avalie múltiplos algoritmos com uma única chamada de função (`train_and_test_classifier`).
* **Pré-processamento Automático**: Inclui escalonamento de features (`StandardScaler`) para modelos sensíveis à escala e codificação one-hot para variáveis categóricas.
* **Ampla Gama de Modelos**: Suporte nativo para classificadores clássicos e ensembles, incluindo Regressão Logística, KNN, SVM, Árvores de Decisão, Random Forest, Redes Neurais (MLP), Bagging, Gradient Boosting, Stacking e um `BlendingClassifier` customizado.
* **Validação Flexível**: Suporte para duas estratégias de validação robustas:
    * **Holdout**: Divisão simples em conjuntos de treino e teste.
    * **K-Fold Cross-Validation**: Validação cruzada estratificada para uma avaliação mais precisa e imparcial do desempenho do modelo.
* **Análise de Desempenho Detalhada**: Geração de um conjunto completo de métricas de classificação, incluindo acurácia, precisão, recall, F1-score, relatório de classificação completo e matriz de confusão.
* **Interpretabilidade de Modelos**: Cálculo da importância das features utilizando métodos configuráveis:
    * **`coef_`**: Para modelos lineares.
    * **Permutation Importance**: Abordagem agnóstica ao modelo que mede a queda de desempenho ao permutar os valores de uma feature.
    * **SHAP (SHapley Additive exPlanations)**: Para uma análise detalhada da contribuição de cada feature para previsões individuais (requer a biblioteca `shap`).

***

## Estrutura do Código

O script é centrado em três componentes principais:

1.  `train_and_test_classifier(...)`: A função principal que orquestra todo o processo, desde a leitura dos dados até a geração do relatório final.
2.  `build_model_pipeline(...)`: Uma função fábrica que constrói e retorna um `Pipeline` do Scikit-learn para um determinado modelo, aplicando o pré-processamento necessário de forma automática.
3.  `BlendingClassifier`: Uma classe customizada que implementa a técnica de ensemble Blending, seguindo a API do Scikit-learn para fácil integração.

***

## Instalação

Para executar o script, as seguintes bibliotecas são necessárias. A biblioteca `shap` é opcional, mas recomendada para análises de interpretabilidade avançadas.

```bash
pip install pandas numpy scikit-learn shap
```

***

## Guia de Uso

A funcionalidade principal é acessada através da função `train_and_test_classifier`. Abaixo, um exemplo de como utilizá-la para treinar um modelo Random Forest com validação cruzada.

```python
from algoritmos_class import train_and_test_classifier

# Configuração da execução do modelo
# Especifique o nome do modelo, caminho do dataset, features, alvo e método de validação.
resultado_rf = train_and_test_classifier(
    model_name='rf',  # 'rf' para Random Forest
    csv_path='path/to/your/dataset.csv',
    feature_columns=['feature1', 'feature2', 'feature_categorica'],
    target_column='alvo',
    validation_method='k_fold',  # Usar validação cruzada de 5 folds
    random_state=42,
    compute_importance=True,
    importance_method='permutation', # Calcular importância via permutação
    model_params={'n_estimators': 150, 'max_depth': 10} # Passar hiperparâmetros
)

# O dicionário 'resultado_rf' contém o modelo final, métricas, importâncias, etc.
print("Métricas médias (K-Fold):", resultado_rf['metrics']['mean'])
print("Importância das features:\n", resultado_rf['feature_importances'].head())

# O modelo final, treinado em todos os dados, está disponível para uso
modelo_final = resultado_rf['model']
# modelo_final.predict(novos_dados)
```

***

## Detalhes Técnicos

### Modelos Suportados

A função `build_model_pipeline` aceita os seguintes identificadores de `model_name`:

| Identificador                               | Modelo                          |
| :------------------------------------------ | :------------------------------ |
| `'knn'`                                     | K-Nearest Neighbors             |
| `'logistic_regression'`, `'logistic'`       | Regressão Logística             |
| `'rf'`, `'random_forest'`                   | Random Forest                   |
| `'svm'`                                     | Support Vector Machine          |
| `'arvore_decisao'`, `'decision_tree'`       | Árvore de Decisão               |
| `'naive_bayes'`, `'nb'`                     | Naive Bayes Gaussiano           |
| `'mlp'`, `'neural_network'`                 | Multi-layer Perceptron          |
| `'ensemble'`                                | Voting Classifier (LR + RF)     |
| `'bagging'`                                 | Bagging com Árvores de Decisão  |
| `'boosting'`                                | Gradient Boosting               |
| `'stacking'`                                | Stacking Ensemble               |
| `'blending'`                                | Blending Ensemble (customizado) |

### Estrutura do Retorno de `train_and_test_classifier`

A função retorna um dicionário detalhado cujo conteúdo depende do `validation_method` escolhido.

#### `validation_method='holdout'`

-   `'model'`: O `Pipeline` treinado no conjunto de treino.
-   `'metrics'`: Dicionário com as métricas (`accuracy`, `precision`, etc.) calculadas no conjunto de teste.
-   `'feature_importances'`: DataFrame do Pandas com a importância de cada feature.
-   `'timings'`: Dicionário com os tempos de execução para `fit`, `predict` e cálculo de importância.
-   `'X-train'`, `'X-test'`, `'y-train'`, `'y-test'`: Os arrays NumPy dos dados de treino e teste.

#### `validation_method='k_fold'`

-   `'model'`: O `Pipeline` final, **treinado com todos os dados**. Este é o modelo pronto para deploy.
-   `'cv_models'`: Uma lista contendo os modelos treinados em cada um dos *k* folds.
-   `'metrics'`: Dicionário aninhado contendo:
    -   `'folds'`: Uma lista de dicionários, cada um com as métricas de um fold.
    -   `'mean'`: A média das métricas de desempenho entre todos os folds.
    -   `'std'`: O desvio padrão das métricas, indicando a variabilidade do desempenho.
-   `'feature_importances'`: DataFrame com a importância calculada usando o modelo final.
-   `'timings'`: Dicionário com tempos de execução detalhados, incluindo tempo médio por fold e tempo de treinamento do modelo final.

### `BlendingClassifier`

Esta é uma implementação customizada da técnica de Blending, que funciona da seguinte forma:

1.  **Divisão de Dados**: O conjunto de treinamento original é dividido em um subconjunto de treino (`train_subset`) e um conjunto de validação (`holdout_set`).
2.  **Treinamento da Primeira Camada**: Os estimadores base (e.g., Random Forest, SVM) são treinados no `train_subset`.
3.  **Geração de Meta-Features**: Os estimadores base treinados são usados para fazer previsões no `holdout_set`. Essas previsões se tornam as *meta-features*.
4.  **Treinamento da Segunda Camada**: Um meta-estimador (e.g., Regressão Logística) é treinado usando as *meta-features* como entrada (`X`) e os rótulos do `holdout_set` como alvo (`y`).
5.  **Re-treinamento (Opcional)**: Para deploy, os estimadores base são re-treinados no conjunto de dados de treinamento completo para maximizar seu poder preditivo. O meta-estimador permanece inalterado.
