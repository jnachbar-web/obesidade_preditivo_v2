import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='Preditor de Obesidade', layout='centered')

st.title('🔬 Sistema Preditivo — Diagnóstico de Obesidade')

st.markdown('Preencha os dados abaixo para prever o nível de obesidade:')


# ===============================
# ✅ Funções para carregar modelos
# ===============================
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load('modelo_obesidade.joblib')
    return modelo

@st.cache_resource
def carregar_label_encoder():
    le_target = joblib.load('labelencoder_obesidade.joblib')
    return le_target

@st.cache_resource
def carregar_features():
    features = joblib.load('features.joblib')
    return features


# ===============================
# ✅ Carregar modelo, labels e features
# ===============================
modelo = carregar_modelo()
le_target = carregar_label_encoder()
features = carregar_features()

rotulos_obesidade_invertido = {v: k for k, v in zip(le_target.classes_, le_target.transform(le_target.classes_))}


# ===============================
# ✅ Interface de inputs
# ===============================
with st.form('form_predicao'):

    st.subheader('⚙️ Dados do paciente')

    genero = st.selectbox('Gênero', ['Feminino', 'Masculino'])
    historico_familiar = st.selectbox('Histórico Familiar de Obesidade', ['Sim', 'Não'])
    consome_calorias = st.selectbox('Consome Alimentos Altamente Calóricos?', ['Sim', 'Não'])
    fuma = st.selectbox('Fuma?', ['Sim', 'Não'])
    monitora_calorias = st.selectbox('Monitora as Calorias?', ['Sim', 'Não'])
    meio_transporte = st.selectbox('Meio de Transporte Predominante', ['Caminhada', 'Bicicleta', 'Transporte Público', 'Automóvel', 'Moto'])

    st.subheader('🔢 Dados Numéricos')

    idade = st.slider('Idade', 10, 100, 30)
    altura = st.number_input('Altura (metros)', 1.20, 2.30, step=0.01, value=1.70)
    peso = st.number_input('Peso (kg)', 30.0, 200.0, step=0.1, value=70.0)

    refeicoes = st.selectbox('Refeições principais por dia', [1, 2, 3, 4])
    agua = st.number_input('Litros de água por dia', 0.0, 5.0, step=0.1, value=1.5)
    tempo_dispositivo = st.number_input('Tempo em dispositivos (horas por dia)', 0.0, 16.0, step=0.5, value=4.0)

    atividade_fisica = st.selectbox('Frequência de Atividade Física', ['Nunca', 'Pouquíssima', 'Moderada', 'Frequente'])

    submit = st.form_submit_button('🔍 Fazer Predição')


# ===============================
# ✅ Processamento da predição
# ===============================
if submit:
    # ✅ Mapeamento dos inputs para numéricos
    mapa_binario = {'Sim': 1, 'Não': 0}
    mapa_genero = {'Masculino': 1, 'Feminino': 0}
    mapa_atividade = {'Nunca': 0, 'Pouquíssima': 1, 'Moderada': 2, 'Frequente': 3}
    mapa_meio_transporte = {'Caminhada': 0, 'Bicicleta': 1, 'Transporte Público': 2, 'Automóvel': 3, 'Moto': 4}

    # ✅ Construir DataFrame com os dados do usuário
    dados = pd.DataFrame([{
        'genero': mapa_genero[genero],
        'historico_familiar': mapa_binario[historico_familiar],
        'consome_alta_calorias_frequente': mapa_binario[consome_calorias],
        'fuma': mapa_binario[fuma],
        'monitora_calorias': mapa_binario[monitora_calorias],
        'meio_transporte_contumaz': mapa_meio_transporte[meio_transporte],

        'idade': idade,
        'altura': altura,
        'peso': peso,
        'qtde_refeicoes_principais': refeicoes,
        'qtde_agua_diaria': agua,
        'tempo_uso_dispositivos': tempo_dispositivo,

        'freq_atividade_fisica': mapa_atividade[atividade_fisica]
    }])

    # ✅ Garantir a ordem e as colunas corretas
    dados = dados[features]

    st.write('Features esperadas:', features)
    st.write('Features recebidas:', list(dados.columns))

    # ✅ Fazer a predição
    pred = modelo.predict(dados)[0]
    resultado = rotulos_obesidade_invertido[pred]

    # ✅ Exibir o resultado
    st.subheader('Resultado da Predição:')
    st.success(f'📊 Nível de Obesidade: **{resultado}**')