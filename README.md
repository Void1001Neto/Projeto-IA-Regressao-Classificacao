# Projeto-IA-Regressao-Classificacao
Análise exploratória, regressão, classificação e otimização com PyCaret e Sklearn.

# 1. Objetivo do Projeto

Este projeto tem como finalidade aplicar técnicas avançadas de Machine Learning em um conjunto de dados fictício sobre o impacto da inteligência artificial no mercado de trabalho até 2030.

O projeto envolve:

1. Análise exploratória detalhada (EDA)
2. Tratamento de dados, detecção de outliers e visualização gráfica
3. Modelagem de Regressão (Linear, Múltipla, Polinomial, PyCaret)
4. Modelagem de Classificação (Naive Bayes, Regressão Logística, PyCaret)
5. Avaliação com métricas apropriadas
6. Otimização via GridSearch, validação cruzada e tuning do PyCaret
7. Criação de dashboard interativo em Jupyter Notebook

########################################################################################


# 2. Estrutura do Repositório

 Projeto-IA-Regressão-Classificação/
│
├── data/
│   └── AI_Impact_on_Jobs_2030.csv
│
├── notebook/
│   └── projeto_final.ipynb
│
├── dashboard/
│   └── dashboard_interativo.ipynb
│
├── src/
│   └── funcoes_auxiliares.py
│
├── requirements.txt
├── LICENSE
└── README.md

########################################################################################


# 3. Descrição do Dataset

O dataset contém informações sobre:

Average_Salary – salário médio estimado

Years_Experience – anos de experiência

AI_Exposure_Index – grau de exposição da profissão à IA

Automation_Probability_2030 – probabilidade de automação

Tech_Growth_Factor – indicador de crescimento tecnológico

Skills_1 a 10 – habilidades avaliadas

Risk_Category – etiqueta de risco (Baixo, Médio, Alto)

* Fonte e licença: Dataset público fictício para uso acadêmico (MIT License).

########################################################################################


# 4. EDA – Análise Exploratória de Dados

Foram utilizados:

* Estatísticas descritivas
* Histogramas e densidades
* Boxplots com normalização
* Scatterplots com jitter e facet grid
* Pairplots estratificados por risco
* Heatmap de correlação incluindo variáveis numéricas e skills

# Principais achados:

A distribuição dos salários é assimétrica e apresenta outliers.

Variáveis de skills foram normalizadas e analisadas em Boxplots.

Probabilidade de automação em 2030 varia fortemente entre categorias de risco.

A correlação entre as variáveis numéricas é baixa, indicando pouca multicolinearidade, exceto em combinações específicas tratadas via VIF.

########################################################################################


# 5. Modelagem
   
# 5.1 Regressão

- Modelos implementados:

- Regressão Linear Simples

- Regressão Linear Múltipla

- Regressão Polinomial

- PyCaret – comparação automática de modelos

- Métricas avaliadas:

- MAE

- RMSE

- R²

- Também foi realizado diagnóstico de resíduos:

- Homocedasticidade

- Normalidade via Q–Q Plot

- Multicolinearidade (VIF)

########################################################################################


# 5.2 Classificação

- Modelos utilizados:

- Naive Bayes

- Regressão Logística

- Gradient Boosting (melhor desempenho)

- PyCaret – ranking de modelos

- Métricas reportadas:

- Accuracy

- Precision

- Recall

- F1-score

- AUC-ROC (macro e micro)

- Matriz de confusão

- O modelo Gradient Boosting obteve desempenho quase perfeito em todas as métricas.

########################################################################################


# 6. Otimização dos Modelos

# PyCaret

compare_models() para ranking inicial

tune_model() para ajuste fino de hiperparâmetros

Seleção automática do melhor modelo

# Sklearn

GridSearchCV para regressão e classificação

Validação cruzada k-fold (k=5)

Comparação de métricas antes e depois da otimização

########################################################################################


 # 7. Dashboard Interativo (Jupyter Notebook)

Inclui:

Distribuições de salário

Scatterplots com densidade

Boxplots normalizados

Heatmap de correlação

Métricas de regressão (sklearn e PyCaret)

Métricas de classificação

Ranking de modelos (classificação e regressão)

Interface construída com:

ipywidgets (Dropdown e interação dinâmica)

Matplotlib/Seaborn

PyCaret outputs integrados

######################################################################################

# 8. Como Executar o Projeto

- Passo 1 — Instalar dependências

pip install -r requirements.txt

- Passo 2 — Abrir o Notebook

No VS Code ou Jupyter:

jupyter notebook

- Passo 3 — Executar o pipeline completo

Execute as células na ordem indicada no notebook principal.

########################################################################################

# 9. Tecnologias Utilizadas

* Python 3.10

* Pandas

* Numpy

* Matplotlib

* Seaborn

* Scikit-Learn

* Statsmodels

* PyCaret

* ipywidgets

########################################################################################

# 10. Licença

Este projeto está sob a licença MIT.
Consulte o arquivo LICENSE para mais informações.

Licença do Dataset
------------------
O arquivo de dados utilizado neste projeto é fictício e foi criado apenas
para fins acadêmicos. O conteúdo está licenciado sob Creative Commons
CC-BY 4.0, permitindo uso e adaptação com atribuição apropriada.

Fonte do Dataset
----------------
Dataset fictício criado para fins educacionais, inspirado em relatórios públicos
sobre automação, mercado de trabalho e impacto da inteligência artificial,
particularmente os estudos do World Economic Forum e OECD AI Policy Observatory.

########################################################################################

# 11. Agradecimentos

Projeto desenvolvido para fins acadêmicos, com foco em aprimoramento de habilidades em:

- Machine Learning aplicado
- Processos completos de EDA
- Implementação e comparação de modelos
- Interpretação estatística
- Construção de dashboards analíticos














