# Projeto Final - Credit Scoring com PyCaret e Streamlit

Este projeto foi desenvolvido como parte do Módulo 38 do curso Profissão Cientista de Dados da EBAC.

O objetivo é construir um modelo de credit scoring para prever a probabilidade de inadimplência de clientes de cartão de crédito. O projeto contempla análise exploratória, tratamento de dados, regressão logística, pipeline com Scikit-learn, avaliação por métricas como Acurácia, AUC, Gini e KS, além de modelagem com PyCaret utilizando LightGBM.

## 🎥 Demonstração da aplicação

A demonstração da aplicação Streamlit em funcionamento pode ser acessada pelo link abaixo:

[streamlit-app-2026-05-18-22-05-29.webm](https://github.com/user-attachments/assets/ecf60b89-0ecb-4179-9007-533b8d12f7bd)


## Arquivos do projeto

- `Mod38Exercicio1.ipynb`: notebook da atividade prática do módulo, com desenvolvimento inicial de credit scoring, pipeline e PyCaret.
- `Mod38Projeto.ipynb`: notebook do projeto final, com o desenvolvimento completo do modelo, avaliação, pipeline, PyCaret e salvamento do modelo.
- `app.py`: aplicação Streamlit para realizar a escoragem de novas bases.
- `model_final.pkl`: modelo final treinado e salvo em pickle.
- `requirements.txt`: bibliotecas necessárias para execução.
- `README.md`: documentação do projeto.

## Etapas desenvolvidas

1. Carregamento da base de dados.
2. Separação das safras de desenvolvimento e out of time.
3. Análise descritiva univariada.
4. Análise descritiva bivariada.
5. Tratamento de valores ausentes.
6. Tratamento de outliers.
7. Criação de variáveis dummies.
8. Treinamento de modelo de regressão logística.
9. Avaliação por Acurácia, AUC, Gini e KS.
10. Criação de pipeline com Scikit-learn.
11. Aplicação de PCA.
12. Modelagem com PyCaret utilizando LightGBM.
13. Salvamento do modelo treinado em pickle.
14. Desenvolvimento de aplicação Streamlit.

## Kernel utilizado

Para executar o notebook completo, principalmente a parte do PyCaret, foi utilizado o kernel:

```text
Python (EBAC PyCaret)
```

Esse kernel foi criado em um ambiente separado para evitar conflitos de dependências com outros projetos.

## Como executar o projeto

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação Streamlit:

```bash
streamlit run app.py
```

## Aplicação

A aplicação permite carregar uma base de clientes, aplicar o modelo treinado e gerar a probabilidade estimada de inadimplência para cada registro.

## Observação

O arquivo original da base de dados não foi incluído no repositório por questões de tamanho e organização. Para executar o notebook completo, é necessário manter a base `credit_scoring.ftr` localmente na mesma pasta do projeto.

O modelo utilizado na aplicação Streamlit é o arquivo `model_final.pkl`, salvo a partir do pipeline de pré-processamento e regressão logística desenvolvido no notebook.

O notebook `Mod38Projeto.ipynb` contempla a modelagem com PyCaret utilizando LightGBM.  
Para a aplicação Streamlit, foi utilizado o arquivo `model_final.pkl`, salvo a partir do pipeline de pré-processamento e regressão logística em Scikit-learn, tornando a aplicação mais leve para execução.

