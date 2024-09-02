import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import google.generativeai as genai
import random
import anthropic

import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import shutil


dotenv.load_dotenv(dotenv.find_dotenv())


google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

openai_models = [
    "gpt-4o", 
    "gpt-4-turbo", 
    "gpt-3.5-turbo-16k", 
    "gpt-4", 
    "gpt-4-32k",
]


# Function to convert the messages format from OpenAI and Streamlit to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages


# Function to query and stream the response from the LLM
def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""

    if model_type == "openai":
        client = OpenAI(api_key=api_key)
        for chunk in client.chat.completions.create(
            model=model_params["model"] if "model" in model_params else "gpt-4o",
            messages=st.session_state.messages,
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
            stream=True,
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name = model_params["model"],
            generation_config={
                "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
            }
        )
        gemini_messages = messages_to_gemini(st.session_state.messages)

        for chunk in model.generate_content(
            contents=gemini_messages,
            stream=True,
        ):
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})


# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:

        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def save_vectorstore(store_name, vectorstore):
    """Función para guardar el VectorStore."""
    try:
        # Verificar si ya existe un archivo o directorio con el mismo nombre
        if os.path.exists(store_name):
            # Eliminar el directorio existente
            shutil.rmtree(store_name)
        vectorstore.save_local(store_name)
        st.write("VectorStore creado y guardado exitosamente.")
    except Exception as e:
        st.write(f"Error al guardar VectorStore: {e}")

def load_vectorstore(store_name, embeddings):
    """Función para cargar el VectorStore desde archivos guardados."""
    try:
        vectorstore = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
        st.write("VectorStore cargado exitosamente.")
    except Exception as e:
        st.write(f"Error al cargar VectorStore: {e}")
        vectorstore = None

    return vectorstore

def main():

    # --- Page Config ---
    st.set_page_config(
        page_title="EMPRESA_CHATBOT",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
    google_api_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") is not None else ""  # only for development environment, otherwise it should return None

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;"><i>Nombre de la empresa</i></h1>""")

    # --- Side Bar ---
    with st.sidebar:
        st.markdown("Puedes poner contenido aquí")  
    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key) and (google_api_key == "" or google_api_key is None):
        st.write("#")
        st.warning("No se encontró ninguna API Key")

    else:
        client = OpenAI(api_key=openai_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # Side bar model options and inputs
        with st.sidebar:

            st.divider()
            
            available_models = [] + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
            # Seleccionar el tipo de respuesta
            respuesta_tipo = st.radio("Elige el tipo de respuesta que deseas:", ("General", "Técnica"))

            # Configurar el modelo según la opción seleccionada
            if respuesta_tipo == "Técnica":
                model="gpt-4o"
            else:
                model="gpt-3.5-turbo"
            #model = st.selectbox("Elige el modelo de IA a utilizar:", available_models, index=0)
            model_type = None
            if model.startswith("gpt"): model_type = "openai"
            elif model.startswith("gemini"): model_type = "google"
            
            with st.popover("⚙️ Parametros del modelo"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reiniciar conversación", 
                on_click=reset_conversation,
            )

            st.divider()

            # Image Upload
            if model in ["gpt-4o", "gpt-4-turbo", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    
                st.write(f"### **🖼️ Sube una imagen{' o un video' if model_type=='google' else ''}:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        if img_type == "video/mp4":
                            # save the video file
                            video_id = random.randint(100000, 999999)
                            with open(f"video_{video_id}.mp4", "wb") as f:
                                f.write(st.session_state.uploaded_img.read())
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "video_file",
                                        "video_file": f"video_{video_id}.mp4",
                                    }]
                                }
                            )
                        else:
                            raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                            img = get_image_base64(raw_img)
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{img_type};base64,{img}"}
                                    }]
                                }
                            )

                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Subir"):
                        st.file_uploader(
                            f"Upload an image{' or a video' if model_type == 'google' else ''}:", 
                            type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []), 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )


            st.divider()
            st.write(f"### **📄 Sube un PDF")
            with st.popover("📁 Subir"):
                pdf = st.file_uploader("Cargar PDF", type=['pdf'])
            if pdf is not None:
                st.write(pdf.name)
                store_name = pdf.name[:-4]

                # Embeddings
                embeddings = OpenAIEmbeddings()

                # Intentar cargar el VectorStore si existe
                vectorstore = load_vectorstore(store_name, embeddings)

                if vectorstore is None:
                    # Si no existe, crearlo desde el PDF
                    pdf_reader = PdfReader(pdf)
                    text = ''
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text=text)

                    # Crear el vectorstore (FAISS)
                    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                    
                    # Guardar el vectorstore para futuras consultas
                    save_vectorstore(store_name, vectorstore)

                # Crear el modelo de chat
                llm = ChatOpenAI(temperature=0)
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)




        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            
            st.session_state.messages.append(
                {
                    "role": "user", 
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                }
            )
                
                # Display the new messages
            with st.chat_message("user"):
                st.markdown(prompt)


            with st.chat_message("assistant"):
                model2key = {
                    "openai": openai_api_key,
                    "google": google_api_key,
                }
                st.write_stream(
                    stream_llm_response(
                        model_params=model_params, 
                        model_type=model_type, 
                        api_key=model2key[model_type]
                    )
                )




if __name__=="__main__":
    main()