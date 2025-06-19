# 🩺 Preditor de Obesidade

Este projeto foi desenvolvido para prever o nível de obesidade com base em dados pessoais, hábitos alimentares e comportamentais.

## 🚀 Como acessar o app
O app está hospedado no Streamlit Cloud: [🔗 Acesse aqui](https://<seu-usuario>-obesidade-preditivo.streamlit.app)

## 📁 Arquivos
- `app.py` → Código do app no Streamlit
- `modelo_obesidade.joblib` → Modelo treinado
- `labelencoder_obesidade.joblib` → Label Encoder para decodificar a saída do modelo
- `requirements.txt` → Dependências do projeto

## 🧠 Modelagem
O modelo final foi treinado utilizando **Gradient Boosting**, com uma acurácia de aproximadamente **95.9%**.

## ✨ Funcionalidades
- Sistema preditivo para diagnóstico de obesidade
- Inputs de variáveis clínicas, alimentares e comportamentais
- Resultado interpretável para o usuário