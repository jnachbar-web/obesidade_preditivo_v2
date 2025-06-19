import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar modelo e label encoder
modelo = joblib.load('modelo_obesidade.joblib')
le_target = joblib.load('labelencoder_obesidade.joblib')

# Dicionário invertido dos labels
rotulos_obesidade_invertido = {v: k for k, v in zip(le_target.classes_, le_target.transform(le_target.classes_))}

st.set_page_config(page_title='Preditor de Obesidade', layout='centered')

st.title('🔬 Sistema Preditivo — Diagnóstico de Obesidade')

st.markdown('Preencha os dados abaixo para prever o nível de obesidade:')

# Sidebar com inputs do usuário
with st.form('form_predicao'):

    genero = st.selectbox('Gênero', ['Feminino', 'Masculino'])
    historico_familiar = st.selectbox('Histórico Familiar de Obesidade', ['Sim', 'Não'])
    consome_calorias = st.selectbox('Consome Alimentos Altamente Calóricos', ['Sim', 'Não'])
    fuma = st.selectbox('Fuma', ['Sim', 'Não'])
    monitora_calorias = st.selectbox('Monitora as Calorias', ['Sim', 'Não'])
    meio_transporte = st.selectbox('Meio de Transporte', ['Caminhada', 'Bicicleta', 'Transporte Público', 'Automóvel', 'Moto'])

    idade = st.slider('Idade', 10, 100, 30)
    altura = st.number_input('Altura (metros)', 1.20, 2.30, step=0.01, value=1.70)
    peso = st.number_input('Peso (kg)', 30.0, 200.0, step=0.1, value=70.0)

    refeicoes = st.selectbox('Refeições por dia', [1, 2, 3, 4])
    agua = st.number_input('Litros de água por dia', 0.0, 5.0, step=0.1, value=1.5)
    tempo_dispositivo = st.number_input('Tempo em dispositivos (horas por dia)', 0.0, 16.0, step=0.5, value=4.0)

    atividade_fisica = st.selectbox('Frequência de Atividade Física', ['Nunca', 'Pouquíssima', 'Moderada', 'Frequente'])

    submit = st.form_submit_button('🔍 Fazer Predição')

# Realizar a predição
if submit:
    # Mapear variáveis categóricas
    mapa_binario = {'Sim': 1, 'Não': 0}
    mapa_genero = {'Masculino': 1, 'Feminino': 0}
    mapa_atividade = {'Nunca': 0, 'Pouquíssima': 1, 'Moderada': 2, 'Frequente': 3}
    mapa_meio_transporte = {'Caminhada': 0, 'Bicicleta': 1, 'Transporte Público': 2, 'Automóvel': 3, 'Moto': 4}

    # Construir DataFrame de entrada
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

    # Fazer a predição
    pred = modelo.predict(dados)[0]
    resultado = rotulos_obesidade_invertido[pred]

    st.subheader('Resultado da Predição:')
    st.success(f'📊 Nível de Obesidade: **{resultado}**')