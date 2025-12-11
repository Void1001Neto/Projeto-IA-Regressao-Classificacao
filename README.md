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

# Fontes usadas na pesquisa 

HUNTER, J. D. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, v. 9, n. 3, p. 90-95, 2007.

MCKINNEY, W. Data Structures for Statistical Computing in Python. In: Proceedings of the 9th Python in Science Conference. Austin, TX, 2010. p. 51-56.

PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011.

SEABOLD, S.; PERKTOLD, J. Statsmodels: Econometric and Statistical Modeling with Python. In: Proceedings of the 9th Python in Science Conference. Austin, TX, 2010.

WASKOM, M. et al. Seaborn: Statistical Data Visualization. Disponível em: https://seaborn.pydata.org
. Acesso em: 10 fev. 2025.

ALI, M. PyCaret: An Open Source, Low-Code Machine Learning Library in Python. Disponível em: https://pycaret.org
. Acesso em: 10 fev. 2025.

MONTGOMERY, D. C.; PECK, E. A.; VINING, G. G. Introduction to Linear Regression Analysis. 5. ed. Hoboken: Wiley, 2012.

DRAPER, N. R.; SMITH, H. Applied Regression Analysis. 3. ed. New York: John Wiley & Sons, 1998.

GUJARATI, D. N.; PORTER, D. C. Econometria Básica. 5. ed. Porto Alegre: AMGH, 2011.

O’BRIEN, R. M. A caution regarding rules of thumb for Variance Inflation Factors. Quality & Quantity, v. 41, p. 673–690, 2007.

MITCHELL, T. M. Machine Learning. New York: McGraw-Hill, 1997.

BISHOP, C. M. Pattern Recognition and Machine Learning. New York: Springer, 2006.

FREUND, Y.; SCHAPIRE, R. E. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. Journal of Computer and System Sciences, v. 55, p. 119–139, 1997.

FRIEDMAN, J. H. Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, v. 29, n. 5, p. 1189–1232, 2001.

FAWCETT, T. An introduction to ROC analysis. Pattern Recognition Letters, v. 27, p. 861–874, 2006.

STONE, M. Cross-validatory choice and assessment of statistical predictions. Journal of the Royal Statistical Society, v. 36, p. 111–147, 1974.

ARLOT, S.; CELISSE, A. A survey of cross-validation procedures for model selection. Statistics Surveys, v. 4, p. 40–79, 2010.

BERGSTRA, J.; BENGIO, Y. Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, v. 13, p. 281–305, 2012.

TUKEY, J. W. Exploratory Data Analysis. Reading: Addison-Wesley, 1977.

########################################################################################

# 11. Agradecimentos

Projeto desenvolvido para fins acadêmicos, com foco em aprimoramento de habilidades em:

- Machine Learning aplicado
- Processos completos de EDA
- Implementação e comparação de modelos
- Interpretação estatística
- Construção de dashboards analíticos














