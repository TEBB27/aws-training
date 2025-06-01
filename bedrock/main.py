from langchain.prompts import PromptTemplate  # Importa la clase para crear plantillas de prompts
from langchain_aws import BedrockLLM         # Importa el modelo BedrockLLM para usar AWS Bedrock


from dotenv import load_dotenv  # Agrega esta línea

import boto3                                 # SDK de AWS para Python, permite crear clientes para servicios AWS
import os                                    # Permite interactuar con el sistema operativo (variables de entorno)
import streamlit as st                       # Framework para crear interfaces web interactivas


load_dotenv(dotenv_path=".env") 
# Configura el perfil de AWS que se usará para las credenciales
os.environ["AWS_PROFILE"] = "219399226095"

# Crea un cliente de boto3 para interactuar con el servicio Bedrock en la región especificada
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Define el ID del modelo que se usará en Bedrock
modelID = "anthropic.claude-v2"

# Inicializa el modelo de lenguaje de Bedrock con los parámetros deseados
llm = BedrockLLM(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 2000, "temperature": 0.9}
)

# Función que genera la respuesta del chatbot usando el modelo y el prompt
def my_chatbot(language, freeform_text):
    # Crea una plantilla de prompt con variables para el idioma y el texto libre
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )
    # Encadena el prompt con el modelo de lenguaje
    chain = prompt | llm
    # Invoca la cadena con los valores proporcionados y obtiene la respuesta
    response = chain.invoke({'language': language, 'freeform_text': freeform_text})
    return response


# if __name__ == "__main__":
#     respuesta = my_chatbot("Spanish", "¿Cómo estás?")
#     print(respuesta)

# --- INTERFAZ DE USUARIO CON STREAMLIT ---

# Crea un menú lateral para seleccionar el idioma
language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Spanish","Portuguese", "French", "German"]
)

# Cambia el título de la app según el idioma seleccionado
if language == "Spanish":
    st.title("Asistente personal de Tomás")
else:
    st.title("My personal assistant")

# Si hay un idioma seleccionado, muestra un área de texto para el mensaje del usuario
if language:
    # freeform_text = st.text_input("Enter your message:")  # Alternativa: campo de texto de una línea
    freeform_text = st.text_area("Enter your message:", height=200, max_chars=50)  # Área de texto multilínea
    # Cuando el usuario presiona el botón "Send", llama a la función del chatbot y muestra la respuesta
    if st.button("Send"):
        response = my_chatbot(language, freeform_text)
        st.write(response)