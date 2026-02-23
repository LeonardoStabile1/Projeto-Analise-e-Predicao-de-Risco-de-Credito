# Risco de Crédito: Análise e Previsões  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Pandas](https://img.shields.io/badge/Lib-Pandas-150458)  
![Scikit Learn](https://img.shields.io/badge/Lib-Scikit_Learn-150458)  
![Status](https://img.shields.io/badge/Status-Finished-success)

# Seção 1: Qual é o problema e por que ele é importante?

No crédito ao consumidor e comercial, as instituições financeiras precisam decidir quais solicitantes provavelmente irão cumprir suas obrigações e quais apresentam maior probabilidade de inadimplência. Uma avaliação imprecisa do risco pode levar ao aumento de perdas de crédito, má alocação de capital e exposição regulatória. Portanto, desenvolver um modelo confiável de previsão de risco de crédito torna-se essencial para apoiar decisões de concessão com critérios objetivos e orientados por dados.

Este projeto aborda esse desafio utilizando o conjunto de dados [Credit Card Approval](https://www.kaggle.com/datasets/youssefelbadry10/credit-card-approval/data) para identificar os principais direcionadores de risco, distinguir solicitantes de baixo risco e estimar a probabilidade de inadimplência por meio de algoritmos supervisionados de Machine Learning. O objetivo é apoiar decisões de aprovação mais consistentes, melhorar a qualidade da carteira e permitir precificação ajustada ao risco com base em evidências quantitativas.

## Principais perguntas:

Antes de prosseguir, definimos duas questões analíticas centrais:

### 1) Quais são os principais direcionadores de risco e quais características estão associadas a menor risco?

Para responder a essa questão, examinamos as variáveis categóricas e estimamos o risco observado dentro de cada categoria. Com base nessas estimativas condicionais, calculamos duas métricas epidemiológicas e de risco padrão: `Diferença de Risco` (RD) e `Risco Relativo` (RR), também chamado de `Lift`.

A Diferença de Risco mede o desvio absoluto em relação ao risco base, enquanto o Risco Relativo mede o desvio proporcional. Formalmente, se $P(Y=y)$ é a probabilidade de um grupo não exposto apresentar o resultado $y$, e $P(Y=y|X=x)$ é a probabilidade do resultado $y$ sob exposição a $X=x$, então:

RD = P(Y = y | X = x) − P(Y = y)
RR = P(Y = y | X = x) / P(Y = y)

Essas métricas permitem identificar categorias que aumentam significativamente (RR > 1) ou reduzem (RR < 1) o risco de inadimplência, tanto em termos absolutos quanto relativos.

### 2) Como o modelo preditivo deve ser avaliado e calibrado?

Como conjuntos de dados de risco de crédito geralmente são desbalanceados, a acurácia global não é uma métrica adequada de desempenho. Em vez disso, priorizamos Recall e F1-score, que capturam melhor o desempenho na classe minoritária (alto risco).

Sob uma perspectiva de teoria da decisão, a calibração do modelo deve refletir o trade-off entre falsos positivos (rejeitar bons clientes) e falsos negativos (aprovar clientes de alto risco). Portanto, a escolha do limiar de decisão deve estar alinhada ao apetite ao risco e à estrutura de custos da instituição.

---

# Seção 2: Análise Exploratória de Dados (EDA)

## Características Gerais

O conjunto de dados contém **36.457 observações** e **20 variáveis**, integrando informações demográficas, socioeconômicas, comportamentais e relacionadas à conta. A variável alvo, **_Is high risk_**, é um indicador binário utilizado para previsão de risco de crédito.

### Grupos de Variáveis

- **Identificador:** `ID`

- **Demográficas:** `Gender`, `Age`, `Marital status`, `Family member count`, `Children count`

- **Socioeconômicas:** `Income`, `Employment status`, `Employment length`, `Education level`, `Job title`

- **Ativos & Acessibilidade:** `Has a car`, `Has a property`, `Phone/email indicators`

- **Habitação:** `Dwelling type`

- **Comportamental / Relacionamento:** `Account age`

- **Alvo:** `Is high risk`

Para uma descrição detalhada do processo de EDA e dos principais insights, consulte o Jupyter Notebook [Credit Card Risk.ipynb](Credit%20Card%20Risk.ipynb).

### Ações de Limpeza de Dados Orientadas pela EDA

- As variáveis `ID` e `Children Count` foram removidas. A primeira não possui relevância analítica e a segunda apresenta alta correlação com `Family member count`, agregando pouca informação incremental.

- As variáveis `Age`, `Employment length` e `Account age` exigiram correção e análise criteriosa de outliers devido a inconsistências identificadas na exploração.

- Pequenos procedimentos de padronização foram aplicados às variáveis `Has a Car` e `Has a property` para garantir consistência categórica.

### Conclusões da Análise Exploratória

- A variável alvo é altamente desbalanceada: apenas **1,7%** das observações são classificadas como alto risco, enquanto **98,3%** são não alto risco. Isso representa um desafio significativo para modelos de Machine Learning, pois há forte tendência de classificar todas as observações como “não alto risco” para maximizar artificialmente a acurácia.

- A Tabela 1 apresenta as categorias mais fortemente associadas ao risco elevado, enquanto a Tabela 2 destaca aquelas associadas a níveis mais baixos de risco relativo. Observa-se que variáveis ocupacionais aparecem com destaque entre os principais direcionadores de risco, sugerindo que a segmentação por emprego desempenha papel central na diferenciação de risco. Certas condições de moradia e estados civis também apresentam risco relativo materialmente maior (RR > 1), indicando efeitos socioeconômicos estruturais.

### Tabela 1: Principais Direcionadores de Risco

| Position | Feature        | Category            | RR (%) |
|----------|---------------|---------------------|--------|
| 1        | Job title     | IT staff            | 296    |
| 2        | Job title     | Low-skill Laborers  | 270    | 
| 3        | Dwelling      | Office apartment    | 203    |
| 4        | Marital status| Widow               | 173    |
| 5        | Dwelling      | Municipal apartment | 158    |

- Por outro lado, as categorias de menor risco concentram-se em determinados status de emprego, níveis educacionais e profissões especializadas. Em particular, maior nível educacional e certas profissões técnicas estão associadas a incidência observada de inadimplência substancialmente menor (RR ≈ 0), reforçando a importância do capital humano e da estabilidade laboral como fatores protetivos.

### Tabela 2: Variáveis Associadas a Menor Nível de Risco

| Position | Feature            | Category                | RR (%) |
|----------|-------------------|--------------------------|--------|
| 51       | Employment status | Student                  | 0      |
| 50       | Education level   | Academic degree          | 0      | 
| 49       | Job title         | Realty agents            | 0      |
| 48       | Job title         | Private service staff    | 0.34   |
| 47       | Job title         | Medicine staff           | 0.48   |

---

# Seção 3: Previsão de Risco com Machine Learning Supervisionado

Após concluir todos os procedimentos de limpeza, diversos algoritmos de Machine Learning do `scikit-learn` foram implementados para simular um cenário real de previsão de risco:

- LogisticRegression  
- RandomForestClassifier  
- LinearSVC  
- DecisionTreeClassifier  
- GradientBoostingClassifier  
- KNeighborsClassifier  

O conjunto de dados foi dividido em **80% para treino** e **20% para teste**. As matrizes de confusão de cada modelo estão disponíveis no Jupyter Notebook. Abaixo, apresentamos as métricas gerais para análise comparativa.

Dado o forte desbalanceamento, a acurácia isoladamente não é métrica adequada. Um modelo ingênuo que previsse `0` (não alto risco) para todas as observações já alcançaria 98,3% de acurácia. Portanto, é necessária avaliação detalhada com métricas mais informativas.

---

# Resultados

## Alto Risco = 0 (Não Alto Risco)

| Model                      | Precision | Recall   | F1-Score |
|----------------------------|-----------|----------|----------|
| LogisticRegression         | 0.9890    | 0.6606   | 0.7924   |
| RandomForestClassifier     | 0.9904    | 0.9662   | 0.9782   |
| DecisionTreeClassifier     | 0.9877    | 0.8843   | 0.9331   |
| GradientBoostingClassifier | 0.9845    | 0.9992   | 0.9917   |
| KNeighborsClassifier       | 0.9857    | 0.9942   | 0.9899   |

Todos os modelos apresentam precisão muito elevada (≥ 0,984), indicando baixa taxa de falsos positivos.

No entanto, LogisticRegression identifica apenas 66,06% dos casos não alto risco, desempenho substancialmente inferior aos demais. GradientBoostingClassifier (99,92%) e KNeighborsClassifier (99,42%) praticamente recuperam toda a classe majoritária.

---

## Alto Risco = 1 (Alto Risco)

A classe Alto Risco representa apenas 1,7% do conjunto de dados, tornando a previsão significativamente mais desafiadora.

| Model                      | Precision | Recall  | F1-Score |
|----------------------------|-----------|---------|----------|
| LogisticRegression         | 0.0277    | 0.5876  | 0.0530   |
| RandomForestClassifier     | 0.1743    | 0.4330  | 0.2485   |
| DecisionTreeClassifier     | 0.0448    | 0.3299  | 0.0790   |
| GradientBoostingClassifier | 0.4444    | 0.0412  | 0.0755   |
| KNeighborsClassifier       | 0.2609    | 0.1237  | 0.1678   |

O comportamento dos modelos difere substancialmente na classe minoritária:

- LogisticRegression apresenta o maior recall (58,76%), mas com precisão extremamente baixa.
- RandomForestClassifier oferece o melhor equilíbrio (maior F1-score = 0,2485).
- GradientBoostingClassifier e KNeighborsClassifier apresentam maior precisão, porém recall muito baixo.
- DecisionTreeClassifier mostra desempenho fraco de forma geral.

No conjunto, RandomForestClassifier oferece o trade-off mais equilibrado para identificar clientes de alto risco.

---

## Desempenho Geral

| Model                      | Accuracy | Weighted Avg Precision | Weighted Avg F1 | Weighted Avg Recall |
|----------------------------|----------|------------------------|-----------------|---------------------|
| LogisticRegression         | 0.6594   | 0.9742                 | 0.7804          | 0.6594              |
| RandomForestClassifier     | 0.9576   | 0.9772                 | 0.9663          | 0.9576              |
| DecisionTreeClassifier     | 0.8753   | 0.9724                 | 0.9193          | 0.8753              |
| GradientBoostingClassifier | 0.9836   | 0.9757                 | 0.9769          | 0.9836              |
| KNeighborsClassifier       | 0.9801   | 0.9739                 | 0.9766          | 0.9801              |

- LogisticRegression apresenta desempenho global fraco.
- DecisionTreeClassifier melhora os resultados, mas permanece limitado na detecção da classe minoritária.
- RandomForestClassifier demonstra desempenho consistente e equilibrado.
- GradientBoostingClassifier alcança a maior acurácia geral, impulsionada pela forte previsão da classe majoritária.
- KNeighborsClassifier apresenta métricas globais sólidas e F1 ponderado ligeiramente superior ao Gradient Boosting.

---

# Conclusão

Devido ao severo desbalanceamento de classes, a avaliação do modelo deve priorizar o trade-off entre precisão e recall, e não apenas a acurácia. Alta acurácia global é fortemente influenciada pela classe dominante e não garante detecção eficaz de clientes de alto risco.

Entre os modelos avaliados, RandomForestClassifier fornece o melhor equilíbrio para a classe minoritária, alcançando o maior F1-score com recall razoável. Ainda assim, nenhum modelo oferece solução totalmente satisfatória sob desbalanceamento extremo.

Uma estratégia mais robusta em contexto real envolveria ajuste de limiar, aprendizado sensível a custo ou estruturas de decisão em ensemble para reduzir falsos negativos e alinhar melhor as previsões ao risco financeiro efetivo.
