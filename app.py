import streamlit as st
import pandas as pd
import joblib

# ================================
# ✅ Configuração da página
# ================================
st.set_page_config(page_title='Preditor de Obesidade', layout='centered')
st.title('🔬 Sistema Preditivo — Diagnóstico de Obesidade')

st.markdown('Preencha os dados abaixo para prever o nível de obesidade:')


# ================================
# ✅ Carregar modelos e dados salvos
# ================================
@st.cache_resource
def carregar_modelo():
    return joblib.load('modelo_obesidade.joblib')

@st.cache_resource
def carregar_label_encoder():
    return joblib.load('labelencoder_obesidade.joblib')

@st.cache_resource
def carregar_features():
    return joblib.load('features.joblib')

# Carregando
modelo = carregar_modelo()
le_target = carregar_label_encoder()
features = carregar_features()

# Dicionário para decodificar os labels
rotulos_obesidade_invertido = {v: k for k, v in zip(le_target.classes_, le_target.transform(le_target.classes_))}


# ================================
# ✅ Formulário de entrada de dados
# ================================
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


# ================================
# ✅ Processamento e Predição
# ================================
if submit:
    # 🔧 Mapeamento das variáveis categóricas e ordinais
    mapa_binario = {'Sim': 1, 'Não': 0}
    mapa_genero = {'Masculino': 1, 'Feminino': 0}
    mapa_atividade = {'Nunca': 0, 'Pouquíssima': 1, 'Moderada': 2, 'Frequente': 3}
    mapa_meio_transporte = {'Caminhada': 0, 'Bicicleta': 1, 'Transporte Público': 2, 'Automóvel': 3, 'Moto': 4}
    mapa_vegetais = {'Nunca ou Raramente': 0, 'Às vezes': 1, 'Sempre': 2}
    mapa_alimentacao = {'Não': 0, 'Às vezes': 1, 'Frequente': 2, 'Sempre': 3}
    mapa_alcool = {'Não': 0, 'Às vezes': 1, 'Frequente': 2, 'Sempre': 3}

    # ✅ Construção do DataFrame de entrada
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

    # ✅ Garantir que a ordem e colunas estejam corretas
    dados = dados[features]

    # 🔍 Validação visual (opcional - pode remover depois)
    with st.expander('🔍 Verificar dados preparados para o modelo'):
        st.write(dados)

    # ✅ Realizar a predição
    pred = modelo.predict(dados)[0]
    resultado = rotulos_obesidade_invertido[pred]

    # ✅ Exibir o resultado
    st.subheader('Resultado da Predição:')
    st.success(f'📊 Nível de Obesidade: **{resultado}**')
