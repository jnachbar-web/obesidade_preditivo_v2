import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração da página
st.set_page_config(page_title='Preditor de Obesidade', layout='wide')

# Função para carregar artefatos
def carregar_artefato(nome_arquivo, descricao):
    try:
        if os.path.exists(nome_arquivo):
            return joblib.load(nome_arquivo)
        else:
            st.error(f'🚫 Arquivo {nome_arquivo} ({descricao}) não encontrado.')
            st.stop()
    except Exception as e:
        st.error(f'❌ Erro ao carregar {descricao}: {e}')
        st.stop()

# Carregar modelo e arquivos auxiliares
modelo = carregar_artefato('modelo_obesidade.joblib', 'Modelo')
label_encoder = carregar_artefato('labelencoder_obesidade.joblib', 'Label Encoder')
features = carregar_artefato('features.joblib', 'Lista de Features')

# Carregar dados
try:
    df = pd.read_csv('Obesity.csv')
except FileNotFoundError:
    st.error('🚫 Arquivo Obesity.csv não encontrado.')
    st.stop()

# Mapeamento de rótulos
ordem_obesidade = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

mapeamento_obesidade = {
    'Insufficient_Weight': 'Abaixo do Peso',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso I',
    'Overweight_Level_II': 'Sobrepeso II',
    'Obesity_Type_I': 'Obesidade I',
    'Obesity_Type_II': 'Obesidade II',
    'Obesity_Type_III': 'Obesidade III'
}

df['Obesity_Label'] = df['Obesity'].map(mapeamento_obesidade)

# ============================
# 📊 Aba 2 — Painel Analítico
# ============================
with aba2:
    st.title('📊 Painel Analítico — Subabas Temáticas')

    subaba = st.selectbox(
        'Selecione a Análise:',
        ['🎯 Distribuição Geral',
         '🔍 Perfil Demográfico',
         '🥦 Estilo de Vida',
         '🔧 Comportamento e Hábitos',
         '🚬 Consumo e Transporte',
         '🔗 Correlação']
    )

    # 🎯 Distribuição Geral
    if subaba == '🎯 Distribuição Geral':
        st.subheader('Distribuição dos Níveis de Obesidade, Peso, Altura e Idade')

        col_esq, col_centro, col_dir = st.columns([1, 2, 1])
        with col_centro:
            fig, ax = plt.subplots(figsize=(6, 4))
            contagem = df['Obesity_Label'].value_counts().reindex(
                [mapeamento_obesidade[k] for k in ordem_obesidade]
            )
            sns.barplot(x=contagem.values, y=contagem.index, color='red', ax=ax)
            ax.set_title('Distribuição dos Níveis de Obesidade', fontsize=12, fontweight='bold')
            ax.set_xlabel('Quantidade', fontsize=9)
            ax.set_ylabel('Nível de Obesidade', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Height'], kde=True, bins=20, color='orange', ax=ax)
            ax.set_title('Distribuição de Altura', fontsize=12, fontweight='bold')
            ax.set_xlabel('Altura (m)', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Weight'], kde=True, bins=20, color='blue', ax=ax)
            ax.set_title('Distribuição de Peso', fontsize=12, fontweight='bold')
            ax.set_xlabel('Peso (kg)', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Age'], kde=True, bins=20, color='green', ax=ax)
            ax.set_title('Distribuição de Idade', fontsize=12, fontweight='bold')
            ax.set_xlabel('Idade (anos)', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

    # 🔍 Perfil Demográfico
    elif subaba == '🔍 Perfil Demográfico':
        st.subheader('Distribuição por Gênero e Histórico Familiar')

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(
                data=df, x='Gender', hue='Obesity_Label',
                palette='Reds', hue_order=[mapeamento_obesidade[k] for k in ordem_obesidade], ax=ax
            )
            ax.set_title('Obesidade por Gênero', fontsize=12, fontweight='bold')
            ax.set_xlabel('Gênero', fontsize=9)
            ax.set_ylabel('Quantidade', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(
                data=df, x='family_history_with_overweight', hue='Obesity_Label',
                palette='Reds', hue_order=[mapeamento_obesidade[k] for k in ordem_obesidade], ax=ax
            )
            ax.set_title('Obesidade x Histórico Familiar', fontsize=12, fontweight='bold')
            ax.set_xlabel('Histórico Familiar', fontsize=9)
            ax.set_ylabel('Quantidade', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

    # 🔗 Correlação
    elif subaba == '🔗 Correlação':
        st.subheader('Mapa de Correlação')

        variaveis_numericas = ['Age', 'Height', 'Weight']
        matriz_correlacao = df[variaveis_numericas].corr()

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matriz_correlacao, annot=True, cmap='Reds', fmt=".2f", ax=ax)
        ax.set_title('Correlação entre Variáveis Numéricas', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
        st.pyplot(fig)

    else:
        st.info('🚧 Subaba em desenvolvimento...')


