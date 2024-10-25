import streamlit as st
from pdf_extractor import PDFExtractor
from llm_model import load_llm_model
import tempfile

# Function to initialize and load the LLM model
def load_llm_model(model_path='phi-2.Q4_K_M.gguf'):
    llm = LLM(
        model_path=model_path,
        n_gpu_layers=40,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        verbose=True,
    )
    return llm

# Class for extracting information from PDF
class PDFExtractor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def extract_text(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=32
        )

        splits = text_splitter.split_documents(data)

        embedding = HuggingFaceEmbeddings()

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
        )

        return vectordb

# UI components
st.title('PDF File and Text Input')

# Upload PDF file
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file is not None:
    pdf_text = PDFExtractor(pdf_file)
    vectordb = pdf_text.extract_text()

    llm = load_llm_model()  # Load LLM model

    # Initialize chat history
    if 'chat_history_prompt' not in st.session_state:
        st.session_state.chat_history_prompt = []

    if 'chat_history_response' not in st.session_state:
        st.session_state.chat_history_response = []

    # Chat input
    messages = st.container()
    with messages:
        user_query = st.text_input("Say something")

        if st.button("Send"):
            if user_query:
                st.session_state.chat_history_prompt.append(user_query)
                for i, query in enumerate(st.session_state.chat_history_prompt):
                    st.write("User Query:", query)
                    response = vectordb.get_relevant_documents(query)[0].page_content

                    llm = load_llm_model()
                    output = llm(
                        f"Instruct: You are an AI assistant for answering questions about the provided context. You are given the following extracted parts of a document. {response} Provide a detailed answer. If you don't know the answer, just say Hmm, I'm not sure. some information about the question {query} Output:",
                        max_tokens=5000,
                        echo=True
                    )

                    st.session_state.chat_history_response.append(output)
                    st.write("Assistant Response:", st.session_state.chat_history_response[i])
