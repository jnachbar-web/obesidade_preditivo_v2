# ============================
# 🚀 Sistema Preditivo + Painel Analítico
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
    if os.path.exists(nome_arquivo):
        return joblib.load(nome_arquivo)
    else:
        st.error(f'🚫 Arquivo {nome_arquivo} ({descricao}) não encontrado.')
        st.stop()

modelo = carregar_artefato('modelo_obesidade.joblib', 'Modelo')
label_encoder = carregar_artefato('labelencoder_obesidade.joblib', 'Label Encoder')
features = carregar_artefato('features.joblib', 'Lista de Features')

# ============================
# 📊 Carregar dados para o Painel Analítico
# ============================
df_graficos = pd.read_csv('Obesity.csv')

# Renomear colunas
df_graficos.rename(columns={
    'Gender':'genero',
    'Age':'idade',
    'Height':'altura',
    'Weight':'peso',
    'family_history_with_overweight':'historico_familiar',
    'FAVC':'consome_alta_calorias_frequente',
    'FCVC':'consumo_vegetais',
    'NCP':'qtde_refeicoes_principais',
    'CAEC':'alimentacao_entre_refeicoes',
    'SMOKE':'fuma',
    'CH2O':'qtde_agua_diaria',
    'SCC':'monitora_calorias',
    'FAF':'freq_atividade_fisica',
    'TUE':'tempo_uso_dispositivos',
    'CALC':'freq_consumo_alcool',
    'MTRANS':'meio_transporte_contumaz',
    'Obesity':'nivel_obesidade'
}, inplace=True)

# Mapeamento de labels
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

df_graficos['Obesity_Label'] = df_graficos['nivel_obesidade'].map(mapeamento_obesidade)

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
        consome_calorias = st.selectbox('Consome alimentos altamente calóricos?', ['Sim', 'Não'])
        consumo_vegetais = st.selectbox('Consome vegetais nas refeições?', ['Nunca ou Raramente', 'Às vezes', 'Sempre'])
        refeicoes = st.selectbox('Refeições principais por dia', [1, 2, 3, 4])
        alimentacao_entre_refeicoes = st.selectbox('Come entre as refeições?', ['Não', 'Às vezes', 'Frequente', 'Sempre'])
        fuma = st.selectbox('Fuma?', ['Sim', 'Não'])
        agua = st.number_input('Litros de água por dia', 0.0, 5.0, step=0.1, value=1.5)
        monitora_calorias = st.selectbox('Monitora as calorias?', ['Sim', 'Não'])
        atividade_fisica = st.selectbox('Frequência de atividade física', ['Nunca', 'Pouquíssima', 'Moderada', 'Frequente'])
        tempo_dispositivo = st.number_input('Tempo em dispositivos (horas/dia)', 0.0, 16.0, step=0.5, value=4.0)
        freq_consumo_alcool = st.selectbox('Consumo de álcool', ['Não', 'Às vezes', 'Frequente', 'Sempre'])
        meio_transporte = st.selectbox('Meio de transporte predominante', ['Caminhada', 'Bicicleta', 'Transporte Público', 'Automóvel', 'Moto'])

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
    st.title('📊 Painel Analítico')

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
            contagem = df_graficos['Obesity_Label'].value_counts().reindex(
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
            sns.histplot(df_graficos['altura'], kde=True, bins=20, color='orange', ax=ax)
            ax.set_title('Distribuição de Altura', fontsize=12, fontweight='bold')
            ax.set_xlabel('Altura (m)', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df_graficos['peso'], kde=True, bins=20, color='blue', ax=ax)
            ax.set_title('Distribuição de Peso', fontsize=12, fontweight='bold')
            ax.set_xlabel('Peso (kg)', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df_graficos['idade'], kde=True, bins=20, color='green', ax=ax)
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
                data=df_graficos, x='genero', hue='Obesity_Label',
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
                data=df_graficos, x='historico_familiar', hue='Obesity_Label',
                palette='Reds', hue_order=[mapeamento_obesidade[k] for k in ordem_obesidade], ax=ax
            )
            ax.set_title('Obesidade x Histórico Familiar', fontsize=12, fontweight='bold')
            ax.set_xlabel('Histórico Familiar', fontsize=9)
            ax.set_ylabel('Quantidade', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig)

    # 🥦 Estilo de Vida
    elif subaba == '🥦 Estilo de Vida':
        st.subheader('Consumo de Vegetais, Atividade Física e Água')

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.boxplot(x='Obesity_Label', y='consumo_vegetais', data=df_graficos,
                        order=[mapeamento_obesidade[k] for k in ordem_obesidade], palette='Reds', ax=ax)
            ax.set_title('Consumo de Vegetais', fontsize=12, fontweight='bold')
            ax.set_xlabel('Nível de Obesidade', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.boxplot(x='Obesity_Label', y='freq_atividade_fisica', data=df_graficos,
                        order=[mapeamento_obesidade[k] for k in ordem_obesidade], palette='Reds', ax=ax)
            ax.set_title('Frequência de Atividade Física', fontsize=12, fontweight='bold')
            ax.set_xlabel('Nível de Obesidade', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.boxplot(x='Obesity_Label', y='qtde_agua_diaria', data=df_graficos,
                        order=[mapeamento_obesidade[k] for k in ordem_obesidade], palette='Reds', ax=ax)
            ax.set_title('Consumo de Água (L)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Nível de Obesidade', fontsize=9)
            ax.set_ylabel('Litros', fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # 🔗 Correlação
    elif subaba == '🔗 Correlação':
        st.subheader('Mapa de Correlação')

        variaveis_numericas = ['idade', 'altura', 'peso', 'qtde_refeicoes_principais', 'qtde_agua_diaria', 'tempo_uso_dispositivos']
        matriz_correlacao = df_graficos[variaveis_numericas].corr()

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matriz_correlacao, annot=True, cmap='Reds', fmt=".2f", ax=ax)
        ax.set_title('Correlação entre Variáveis Numéricas', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
        st.pyplot(fig)

    else:
        st.info('🚧 Esta subaba será construída na próxima etapa...')