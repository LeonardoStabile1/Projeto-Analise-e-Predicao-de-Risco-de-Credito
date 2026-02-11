# Análise de dados Exploratória (EDA) - Visão geral

Antes de começar, precisamos conhecer cada uma das features do dataset:
- Dados numéricos: `Age`,`Family member count`, `Account age`, `Income`, `Employment length`
- Dados categóricos: `Has a mobile phone`, `Has a work phone`, `Has a phone`, `Has a car`, `Has a property`, `Has an email`, `Gender`, `Employment status`, `Marital status`, `Dwelling`, `Job title`

As features `Has a car` e `Has a property` foram mapeadas em {0,1} ao invés de {Y,N}, enquanto as colunas `Age`, `Account age` e `Employment length` precisaram ser convertidas para anos. Por fim, os resultados `NaN` presentes na feature `Job title` foram renomeados para `no title`, pois esta ausencia não implica na remoção dos dados.

## Feature alvo: `Is high risk`

Os dados exibem uma enorme tendencia aos casos de *sem risco*. Este distribuição desigual complica a predição de padrões de alto risco no caso de Machine Learning.

![](figures_eda/is_high_risk.png)

## Dados categóricos: distribuição

![](figures_eda/gender_frequency.png)
![](figures_eda/employment_stats_frequency.png)
![](figures_eda/education_level.png)
![](figures_eda/marital_status.png)
![](figures_eda/dwelling_frequency.png)
![](figures_eda/job_title_frequency.png)
![](figures_eda/has_a_car_frequency.png)
![](figures_eda/has_a_property_frequency.png)
![](figures_eda/has_a_work_phone_frequency.png)
![](figures_eda/has_a_phone_frequency.png)
![](figures_eda/has_an_email_frequency.png)

## Dados numéricos: distribuição

Antes de visualizar cada uma das colunas, vamos visualizar a matriz de correlação com o objetivo de identificar features que estão correlacionadas entre si

![](figures_eda/correlation_matrix.png)

Vemos resultados importantes:

- `Family member count` e `Children count` apresentam uma grande correlação, justificada pela natureza das features.
- `Age` e `Employment length` também apresentam uma correlação considerável, mas que não necessariamente implicam uma na outra.
- `Age` e `Children count` também apresentam uma correlação.
- `Age` and `Family member count`.

Devido a multicolinearidade, vou remover a coluna `Children count` nesta analise, já que o `Family member count` já apresenta um bom indicativo.

![](figures_eda/Income_bp.png)
![](figures_eda/Age_bp.png)

A feature `Employment length` possui um outliner claro: com valor maior que 1000 anos, esta entrada representa um dado errado/faltante. Portanto, este será removido da análise.

![](figures_eda/employment_length_bp.png)
![](figures_eda/family_members_bp.png)
![](figures_eda/account_age_bp.png)

Da análise por meio dos diagramas de caixa, vemos que `Income`, `Employment length` e `Family member count` posui outliers naturais. Esses dados foram classificados e tratados como reais e possíveis de acontecer, representando variações socioeconomicas ao invés de entradas erradas.

## Coomentários

A análise preliminar sugere que nenhuma feature age como central na determinação da variável `Is high risk`. Portanto, seguimos com o dataset preparado para implementação de Machine Learning.
