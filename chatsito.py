import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import shutil

# Sidebar
with st.sidebar:
    st.title('Chat-PDF clone :D')
    st.markdown('''
                ## Acerca de
                Esta app  está potenciada con un LLM (Large Language Model) de OpenAI usando:
                - [Streamlit](https://www.streamlit.io/)
                - [OpenAI](https://www.openai.com/)
                - [Langchain](https://python.langchain.com/)
                ''')
    add_vertical_space(5)
    st.write('Hecho con amor')

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


load_dotenv()

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
    st.title('Bienvenido al chat inteligente!')
    st.header('Bienvenido al chat inteligente!')
    st.write('Hecho con amor')

    # Cargar el PDF
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

        # Interfaz de chat
        st.write("Puedes hacer preguntas sobre el documento:")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        user_question = st.text_input("Haz una pregunta sobre el documento:")
        if user_question:
            response = qa_chain({"question": user_question, "chat_history": st.session_state['chat_history']})
            st.session_state['chat_history'].append((user_question, response['answer']))
            st.write(response['answer'])

        # Mostrar el historial de chat
        if st.session_state['chat_history']:
            st.write("Historial de chat:")
            for i, (q, a) in enumerate(st.session_state['chat_history']):
                st.write(f"**Pregunta {i + 1}:** {q}")
                st.write(f"**Respuesta {i + 1}:** {a}")

if __name__ == '__main__':
    main()
