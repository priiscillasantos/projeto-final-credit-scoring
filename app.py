import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# Configuração da página
st.set_page_config(
    page_title='Credit Scoring - Módulo 38',
    layout='wide'
)


# Carregar o modelo salvo
@st.cache_resource
def carregar_modelo():
    with open('model_final.pkl', 'rb') as arquivo:
        modelo = pickle.load(arquivo)
    return modelo


modelo = carregar_modelo()


# Título principal
st.title('Credit Scoring - Módulo 38')

st.write(
    'Aplicação desenvolvida para escorar uma base de clientes utilizando '
    'o modelo treinado no projeto final.'
)


# Sidebar
st.sidebar.header('Configurações')

ponto_corte = st.sidebar.slider(
    'Ponto de corte',
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

st.sidebar.write(
    'Clientes com probabilidade maior ou igual ao ponto de corte '
    'serão classificados como **Mau**.'
)


# Upload da base
st.subheader('Upload da base')

arquivo = st.file_uploader(
    'Suba um arquivo CSV para escoragem',
    type=['csv']
)


if arquivo is not None:

    base = pd.read_csv(arquivo)

    st.success('Arquivo carregado com sucesso.')

    st.subheader('Prévia da base enviada')
    st.dataframe(base.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric('Quantidade de linhas', base.shape[0])

    with col2:
        st.metric('Quantidade de colunas', base.shape[1])

    # Removendo colunas que não devem ser usadas pelo modelo, caso existam
    colunas_remover = ['mau', 'data_ref', 'index']
    base_modelo = base.drop(columns=colunas_remover, errors='ignore').copy()

    st.info(
        'Caso a base contenha as colunas `mau`, `data_ref` ou `index`, '
        'elas serão removidas automaticamente antes da escoragem.'
    )

    if st.button('Escorar base'):

        try:
            # Gerar probabilidade da classe 1, que representa mau pagador
            probabilidades = modelo.predict_proba(base_modelo)[:, 1]

            resultado = base.copy()
            resultado['probabilidade_mau'] = probabilidades
            resultado['classificacao'] = resultado['probabilidade_mau'].apply(
                lambda x: 'Mau' if x >= ponto_corte else 'Bom'
            )

            st.subheader('Resumo da escoragem')

            qtd_bom = (resultado['classificacao'] == 'Bom').sum()
            qtd_mau = (resultado['classificacao'] == 'Mau').sum()
            perc_mau = qtd_mau / len(resultado) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric('Clientes classificados como Bom', qtd_bom)

            with col2:
                st.metric('Clientes classificados como Mau', qtd_mau)

            with col3:
                st.metric('% de clientes Mau', f'{perc_mau:.2f}%')

            st.subheader('Resultado da escoragem')
            st.dataframe(resultado.head(20))

            # Gráfico de classificação
            st.subheader('Distribuição das classificações')

            contagem_classificacao = resultado['classificacao'].value_counts()

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(contagem_classificacao.index, contagem_classificacao.values)
            ax.set_title('Quantidade de clientes por classificação')
            ax.set_xlabel('Classificação')
            ax.set_ylabel('Quantidade de clientes')

            st.pyplot(fig)

            # Gráfico de probabilidade
            st.subheader('Distribuição das probabilidades de inadimplência')

            fig2, ax2 = plt.subplots(figsize=(9, 4))
            ax2.hist(resultado['probabilidade_mau'], bins=20)
            ax2.axvline(ponto_corte, linestyle='--')
            ax2.set_title('Distribuição da probabilidade de mau pagador')
            ax2.set_xlabel('Probabilidade de mau')
            ax2.set_ylabel('Quantidade de clientes')

            st.pyplot(fig2)

            # Filtro dinâmico
            st.subheader('Filtro por classificação')

            filtro = st.selectbox(
                'Selecione a classificação para visualizar',
                ['Todos', 'Bom', 'Mau']
            )

            if filtro == 'Todos':
                resultado_filtrado = resultado.copy()
            else:
                resultado_filtrado = resultado[resultado['classificacao'] == filtro].copy()

            st.dataframe(resultado_filtrado)

            # Clientes de maior risco
            st.subheader('Top 20 clientes com maior risco')

            clientes_risco = resultado.sort_values(
                by='probabilidade_mau',
                ascending=False
            ).head(20)

            st.dataframe(clientes_risco)

            # Download
            csv_resultado = resultado.to_csv(index=False).encode('utf-8')

            st.download_button(
                label='Baixar base escorada em CSV',
                data=csv_resultado,
                file_name='base_escorada.csv',
                mime='text/csv'
            )

        except Exception as erro:
            st.error('Ocorreu um erro ao escorar a base.')
            st.write(erro)

else:
    st.warning('Faça o upload de um arquivo CSV para iniciar a escoragem.')