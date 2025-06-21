# ============================
# 🚀 Sistema Preditivo + Painel Analítico (Otimizado)
# ============================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================
# 🎨 Configuração da Página
# ============================
st.set_page_config(page_title='Preditor de Obesidade', layout='wide')


# ============================
# 📂 Carregar Artefatos
# ============================
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


modelo = carregar_artefato('modelo_obesidade.joblib', 'Modelo')
label_encoder = carregar_artefato('labelencoder_obesidade.joblib', 'Label Encoder')
features = carregar_artefato('features.joblib', 'Lista de Features')

try:
    df = pd.read_csv('Obesity.csv')
except FileNotFoundError:
    st.error('🚫 Arquivo Obesity.csv não encontrado.')
    st.stop()


# ============================
# 🔠 Mapeamento de Labels
# ============================
ordem_obesidade = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
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
# 🔀 Abas do App
# ============================
aba1, aba2 = st.tabs(['🔍 Sistema Preditivo', '📊 Painel Analítico'])


# ============================
# 🔍 Aba 1 — Sistema Preditivo
# ============================
with aba1:
    st.title('🔍 Sistema Preditivo — Diagnóstico de Obesidade')

    with st.form('form_predicao'):
        st.subheader('⚙️ Dados Gerais')

        genero = st.selectbox('Gênero', ['Feminino', 'Masculino'])
        idade = st.slider('Idade', 10, 100, 30)
        altura = st.number_input('Altura (metros)', 1.20, 2.30, step=0.01, value=1.70)
        peso = st.number_input('Peso (kg)', 30.0, 200.0, step=0.1, value=70.0)

        st.subheader('⚙️ Hábitos e Estilo de Vida')

        historico_familiar = st.selectbox('Histórico Familiar de Obesidade', ['Sim', 'Não'])
        consome_calorias = st.selectbox('Consome Alimentos Altamente Calóricos?', ['Sim', 'Não'])
        consumo_vegetais = st.selectbox('Você costuma comer vegetais nas suas refeições?', ['Nunca ou Raramente', 'Às vezes', 'Sempre'])
        refeicoes = st.selectbox('Refeições principais por dia', [1, 2, 3, 4])
        alimentacao_entre_refeicoes = st.selectbox('Você come alguma coisa entre as refeições?', ['Não', 'Às vezes', 'Frequente', 'Sempre'])
        fuma = st.selectbox('Fuma?', ['Sim', 'Não'])
        agua = st.number_input('Litros de água por dia', 0.0, 5.0, step=0.1, value=1.5)
        monitora_calorias = st.selectbox('Monitora as Calorias?', ['Sim', 'Não'])
        atividade_fisica = st.selectbox('Frequência de Atividade Física', ['Nunca', 'Pouquíssima', 'Moderada', 'Frequente'])
        tempo_dispositivo = st.number_input('Tempo em dispositivos (horas por dia)', 0.0, 16.0, step=0.5, value=4.0)
        freq_consumo_alcool = st.selectbox('Com que frequência você bebe álcool?', ['Não', 'Às vezes', 'Frequente', 'Sempre'])
        meio_transporte = st.selectbox('Meio de Transporte Predominante', ['Caminhada', 'Bicicleta', 'Transporte Público', 'Automóvel', 'Moto'])

        submit = st.form_submit_button('🔍 Fazer Predição')


    if submit:
        mapa_binario = {'Sim': 1, 'Não': 0}
        mapa_genero = {'Masculino': 1, 'Feminino': 0}
        mapa_atividade = {'Nunca': 0, 'Pouquíssima': 1, 'Moderada': 2, 'Frequente': 3}
        mapa_meio_transporte = {'Caminhada': 0, 'Bicicleta': 1, 'Transporte Público': 2, 'Automóvel': 3, 'Moto': 4}
        mapa_vegetais = {'Nunca ou Raramente': 0, 'Às vezes': 1, 'Sempre': 2}
        mapa_alimentacao = {'Não': 0, 'Às vezes': 1, 'Frequente': 2, 'Sempre': 3}
        mapa_alcool = {'Não': 0, 'Às vezes': 1, 'Frequente': 2, 'Sempre': 3}

        dados = pd.DataFrame([{
            'genero': mapa_genero[genero],
            'idade': idade,
            'altura': altura,
            'peso': peso,
            'historico_familiar': mapa_binario[historico_familiar],
            'consome_alta_calorias_frequente': mapa_binario[consome_calorias],
            'consumo_vegetais': mapa_vegetais[consumo_vegetais],
            'qtde_refeicoes_principais': refeicoes,
            'alimentacao_entre_refeicoes': mapa_alimentacao[alimentacao_entre_refeicoes],
            'fuma': mapa_binario[fuma],
            'qtde_agua_diaria': agua,
            'monitora_calorias': mapa_binario[monitora_calorias],
            'freq_atividade_fisica': mapa_atividade[atividade_fisica],
            'tempo_uso_dispositivos': tempo_dispositivo,
            'freq_consumo_alcool': mapa_alcool[freq_consumo_alcool],
            'meio_transporte_contumaz': mapa_meio_transporte[meio_transporte]
        }])

        dados = dados[features]
        pred = modelo.predict(dados)[0]
        resultado = label_encoder.inverse_transform([pred])[0]

        st.subheader('Resultado da Predição:')
        st.success(f'📊 Nível de Obesidade: **{resultado}**')


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

    if subaba == '🎯 Distribuição Geral':
        st.subheader('Distribuição dos Níveis de Obesidade, Peso, Altura e Idade')

        # 🔺 Gráfico — Níveis de Obesidade (Centralizado)
        col_esq, col_centro, col_dir = st.columns([1, 2, 1])

        with col_centro:
            fig, ax = plt.subplots(figsize=(6, 4))
            contagem = df['Obesity_Label'].value_counts().reindex(
                [mapeamento_obesidade[k] for k in ordem_obesidade]
            )
            sns.barplot(x=contagem.values, y=contagem.index, color='red', ax=ax)
            ax.set_title('Distribuição dos Níveis de Obesidade')
            ax.set_xlabel('Quantidade')
            ax.set_ylabel('Nível de Obesidade')
            st.pyplot(fig)

        # 🔻 Três gráficos lado a lado
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Height'], kde=True, bins=20, color='orange', ax=ax)
            ax.set_title('Distribuição de Altura')
            ax.set_xlabel('Altura (m)')
            ax.set_ylabel('Frequência')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Weight'], kde=True, bins=20, color='blue', ax=ax)
            ax.set_title('Distribuição de Peso')
            ax.set_xlabel('Peso (kg)')
            ax.set_ylabel('Frequência')
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['Age'], kde=True, bins=20, color='green', ax=ax)
            ax.set_title('Distribuição de Idade')
            ax.set_xlabel('Idade (anos)')
            ax.set_ylabel('Frequência')
            st.pyplot(fig)

    elif subaba == '🔍 Perfil Demográfico':
        st.subheader('Distribuição por Gênero e Histórico Familiar')
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(data=df, x='Gender', hue='Obesity_Label', palette='Reds', hue_order=[mapeamento_obesidade[k] for k in ordem_obesidade], ax=ax)
            ax.set_title('Obesidade por Gênero')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(data=df, x='family_history_with_overweight', hue='Obesity_Label', palette='Reds', hue_order=[mapeamento_obesidade[k] for k in ordem_obesidade], ax=ax)
            ax.set_title('Obesidade x Histórico Familiar')
            st.pyplot(fig)

    elif subaba == '🔗 Correlação':
        st.subheader('Mapa de Correlação')
        variaveis_numericas = ['Age', 'Height', 'Weight']
        matriz_correlacao = df[variaveis_numericas].corr()

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matriz_correlacao, annot=True, cmap='Reds', fmt=".2f", ax=ax)
        ax.set_title('Correlação entre Variáveis Numéricas')
        st.pyplot(fig)

    else:
        st.info('🚧 Subaba em desenvolvimento...')
