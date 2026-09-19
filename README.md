# 💳 Detecção de Fraude em Transações com Machine Learning

Projeto de Machine Learning desenvolvido para identificar transações potencialmente fraudulentas utilizando diferentes algoritmos de classificação e técnicas de avaliação e interpretabilidade de modelos.

## 🎯 Objetivo

Desenvolver e comparar modelos de classificação capazes de identificar transações fraudulentas em um conjunto de dados de operações com cartão de crédito.

O projeto também explora técnicas para lidar com o desbalanceamento das classes, ajuste de threshold, otimização de hiperparâmetros e interpretação das previsões dos modelos.

## 🛠️ Tecnologias e Bibliotecas

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Google Colab

## 📊 Dataset

O projeto utiliza o dataset **Credit Card Fraud Detection**, disponibilizado pelo TensorFlow.

Os dados são carregados diretamente através da URL:

```text
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

O conjunto de dados contém transações de cartão de crédito classificadas entre operações legítimas e fraudulentas.

## 🔎 Etapas do Projeto

### 1. Carregamento e exploração dos dados

Inicialmente, os dados são carregados utilizando Pandas e são analisadas as primeiras observações e a proporção entre as classes.

### 2. Feature Engineering

Foram criadas novas variáveis a partir da coluna `Amount`:

* `Amount_log`: transformação logarítmica utilizando `log1p`;
* `Amount_scaled`: padronização do valor da transação utilizando `StandardScaler`.

### 3. Separação entre treino e teste

Os dados foram divididos em conjuntos de treinamento e teste utilizando `train_test_split`, mantendo a proporção das classes através de `stratify`.

### 4. Regressão Logística

Foi desenvolvido um modelo de Regressão Logística utilizando um Pipeline com padronização dos dados.

Também foi realizado um experimento alterando o threshold de classificação de `0.5` para `0.3`, permitindo analisar o impacto dessa alteração nas métricas de classificação.

### 5. Random Forest

Foi utilizado um modelo Random Forest com:

* `class_weight="balanced"`
* 50 árvores
* profundidade máxima de 10

Essa configuração foi utilizada considerando o desbalanceamento existente entre as classes.

### 6. Avaliação com ROC e AUC

O desempenho do Random Forest foi analisado através da:

* Curva ROC;
* Área sob a curva ROC (AUC).

### 7. XGBoost

Foi utilizado o XGBoost como modelo mais avançado para classificação das transações.

O modelo foi treinado considerando o desbalanceamento das classes através do parâmetro `scale_pos_weight`.

### 8. Importância das Variáveis

Após o treinamento do XGBoost, foi analisada a importância das variáveis utilizadas pelo modelo.

O gráfico apresenta as 10 variáveis com maior importância para as decisões do algoritmo.

### 9. Otimização com GridSearchCV

Foi utilizado `GridSearchCV` para testar diferentes combinações de hiperparâmetros do XGBoost.

Foram avaliadas diferentes configurações de:

* `max_depth`;
* `n_estimators`.

A métrica utilizada para seleção dos melhores parâmetros foi o **recall**, considerando o objetivo de identificar corretamente as transações fraudulentas.

### 10. Interpretabilidade com SHAP

Por fim, foi utilizada a biblioteca SHAP para analisar como as variáveis influenciam as previsões do modelo otimizado.

Essa etapa permite obter uma visão mais interpretável do comportamento do modelo de Machine Learning.

## 📈 Modelos Utilizados

| Modelo                 | Objetivo             |
| ---------------------- | -------------------- |
| Regressão Logística    | Modelo base          |
| Random Forest          | Modelo intermediário |
| XGBoost                | Modelo avançado      |
| XGBoost + GridSearchCV | Otimização           |
| SHAP                   | Interpretabilidade   |

## 📂 Estrutura do Projeto

```text
deteccao-fraude-machine-learning/
│
├── notebooks/
│   └── deteccao_fraude.ipynb
│
├── README.md
│
└── requirements.txt
```

## ▶️ Como executar

### Google Colab

O notebook pode ser executado diretamente no Google Colab.

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLE_AQUI_O_LINK_DO_COLAB)

### Localmente

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois abra o notebook:

```bash
jupyter notebook
```

## 📚 Principais conceitos praticados

* Classificação supervisionada
* Feature Engineering
* Tratamento de dados desbalanceados
* Train/Test Split
* Threshold de classificação
* Random Forest
* XGBoost
* ROC e AUC
* Recall
* Otimização de hiperparâmetros
* GridSearchCV
* Interpretabilidade de Machine Learning
* SHAP

## 👨‍💻 Autor

**Deyvison Vitor**

Projeto desenvolvido como parte da minha formação prática em desenvolvimento e análise de dados utilizando Python e Machine Learning.
