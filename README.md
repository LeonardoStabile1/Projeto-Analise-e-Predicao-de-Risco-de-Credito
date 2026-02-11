# Risco de crédito: Análise e Predição

## Contexto

Este projeto utiliza o dataset [`Credit Card Approval`](https://www.kaggle.com/datasets/youssefelbadry10/credit-card-approval/data) disponível no Kaggle para entender, filtrar e limpar os dados com um estudo exploratório (EDA). Por fim, cálculos de predição de alto e baixo risco são realizados por meio de diversos métodos de machine learning.

---

## Estudo exploratório (EDA)

### Visão geral dos dados

O conjunto de dados contém 29.165 observações e 20 variáveis, combinando informações demográficas, socioeconômicas, comportamentais e relacionadas à conta. A variável alvo é Is high risk, um indicador binário utilizado para previsão de risco de crédito.

Grupos de Variáveis:

- Identificador: ID

- Demográficas: Gender, Age, Marital status, Family member count, Children Count

- Socioeconômicas: Income, Employment status, Employment length, Education level, Job title

- Ativos & Acessibilidade: Has a car, Has a property, phone/email indicators

- Habitação: Dwelling

- Comportamental / Relacionamento: Account age

- Alvo: Is high risk

Para uma análise detalhada da fase de exploração dos dados e dos principais insights, consulte o [arquivo Jupyter](Credit%20Card%20Risk.ipynb), e para os resultados desta análise, o arquivo [EDA](EDA.md).

### Conclusões do estudo exploratório
- A variável alvo é altamente desbalanceada: apenas 1,7% do total dos dados está classificado como alto risco, enquanto 98,3% não representa situação de alto risco. Isso constitui um grande desafio para modelos de Machine Learning, pois há uma forte tendência de classificar todas as observações como “não alto risco” para maximizar a acurácia aparente;

- Nenhuma variável isoladamente apresenta forte poder preditivo de risco; portanto, espera-se que a capacidade preditiva esteja principalmente nas interações entre variáveis (feature interactions).

### Ações de Limpeza Orientadas pela EDA

- Algumas variáveis, como `ID` e `Children Count`, foram removidas do dataframe, pois a primeira não possui relevância analítica e a segunda apresenta alta correlação com `Family member count`.

- As variáveis `Age`, `Employment length` e `Account age` precisaram ser corrigidas e analisadas cuidadosamente para identificação de outliers que representavam inconsistências nos dados.

- Foram realizadas pequenas padronizações nas variáveis `Has a Car` e `Has a property`.

# Abordagem de Machine Learning Supervisionado para Previsão de Risco de Crédito

Após todas as etapas de limpeza, alguns métodos de machine learning do `scikit-learn` foram utilizados para simular um cenário real de previsão:

- LogisticRegression  
- RandomForestClassifier  
- LinearSVC  
- DecisionTreeClassifier  
- GradientBoostingClassifier  
- KNeighborsClassifier  

Os dados foram divididos em 80% para treino e 20% para teste. As matrizes de confusão de cada método podem ser observadas no notebook Jupyter. Aqui, apresento os resultados gerais de cada modelo para análise comparativa.  

Como os dados são altamente desbalanceados, a acurácia não é uma métrica adequada isoladamente, pois um modelo que preveja `0` para todas as observações alcançaria 98,3% de acurácia. Portanto, é necessário analisar as métricas detalhadas apresentadas nas tabelas a seguir.

---

## High-Risk = 0 (Não alto risco)

| Model                          | Precision | Recall   | F1-Score | 
|--------------------------------|-----------|----------|----------|
| LogisticRegression             | 0.9890    | 0.6419   | 0.7785   |
| RandomForestClassifier         | 0.9890    | 0.9407   | 0.9642   |
| DecisionTreeClassifier         | 0.9882    | 0.8963   | 0.9400   |
| GradientBoostingClassifier     | 0.9847    | 0.9992   | 0.9919   | 
| KNeighborsClassifier           | 0.9861    | 0.9969   | 0.9915   |

Observa-se que todos os modelos apresentam alta precisão, com aproximadamente 1% a 1,5% de falsos positivos para a classe de não alto risco. A métrica de Recall indica que, entre as 4.772 observações da classe 0, o LogisticRegression identificou apenas 64%, desempenho significativamente inferior aos demais modelos. O F1-Score, que representa o equilíbrio entre Precision e Recall, confirma que o LogisticRegression não é a melhor alternativa para essa classe.

---

## High-Risk = 1 (Alto risco)

A classe High-Risk = 1 representa apenas 1,7% do total de dados, o que torna sua previsão significativamente mais complexa.

| Model                          | Precision | Recall  | F1-Score |
|--------------------------------|-----------|---------|----------|
| LogisticRegression             | 0.0262    | 0.5750  | 0.0501   |
| RandomForestClassifier         | 0.0958    | 0.3750  | 0.1527   |
| DecisionTreeClassifier         | 0.0553    | 0.3625  | 0.0960   |
| GradientBoostingClassifier     | 0.6000    | 0.0750  | 0.1333   |
| KNeighborsClassifier           | 0.4643    | 0.1625  | 0.2407   |


Analisando os algoritmos, notamos que oGradientBoostingClassifier e o KNeighborsClassifier apresentam alta precisão, mas recall muito baixo — ou seja, identificam poucos casos de alto risco. O LogisticRegression conseguiu identificar 57,5% dos casos de alto risco (maior recall), porém com precisão extremamente baixa. O RandomForestClassifier apresentou um desempenho mais equilibrado, refletido em um F1-Score superior aos demais modelos para essa classe.

---

## Desempenho Geral

| Model                          | Accuracy  | Weighted Avg Precision | Weighted Avg F1 | Weighted Avg Recall |
|--------------------------------|-----------|------------------------|-----------------|---------------------|
| LogisticRegression             | 0.6408    | 0.9731                 | 0.7665          | 0.6408              |
| RandomForestClassifier         | 0.9314    | 0.9743                 | 0.9509          | 0.9314              |
| DecisionTreeClassifier         | 0.8875    | 0.9728                 | 0.9261          | 0.8875              |
| GradientBoostingClassifier     | 0.9839    | 0.9784                 | 0.9777          | 0.9839              |
| KNeighborsClassifier           | 0.9831    | 0.9775                 | 0.9791          | 0.9831              |

- O LogisticRegression apresenta boa precisão na classe majoritária, mas baixo recall, resultando em F1-Score inferior.  
- O DecisionTreeClassifier melhora o desempenho geral em relação à regressão logística, mas ainda mantém recall moderado.  
- O RandomForestClassifier demonstra desempenho consistente e equilibrado entre precisão e recall, lidando melhor com o desbalanceamento devido ao uso de ensemble.  
- O GradientBoostingClassifier alcança a maior acurácia geral e excelente F1 ponderado, mas ainda apresenta dificuldades na classe minoritária.  
- O KNeighborsClassifier também apresenta desempenho global elevado e F1 ponderado ligeiramente superior ao Gradient Boosting, embora possa ser sensível à distribuição da classe minoritária.

---

# Conclusão

Dado o forte desbalanceamento das classes, a avaliação dos modelos deve priorizar o trade-off entre precision e recall, e não apenas a acurácia. Os resultados demonstram que modelos com alta acurácia global tendem a acertar majoritariamente a classe dominante, enquanto ainda enfrentam dificuldades para identificar consistentemente os casos de alto risco. **Nenhum modelo isoladamente apresenta solução ideal.** Alguns priorizam recall para a classe minoritária com perda de precisão, enquanto outros são altamente precisos, mas deixam de identificar parcela relevante dos casos de alto risco. Isso evidencia uma limitação estrutural de classificadores únicos em cenários severamente desbalanceados. Portanto, uma abordagem combinada utilizando os modelos com melhor desempenho é mais recomendada do que a escolha de um único algoritmo. Essa estratégia permite validação cruzada das previsões, reduz o risco de falsos negativos em casos de alto risco e fornece uma estrutura decisória mais robusta e confiável para aplicação em contexto real, onde a não identificação de um cliente de alto risco pode gerar impactos financeiros significativos.
