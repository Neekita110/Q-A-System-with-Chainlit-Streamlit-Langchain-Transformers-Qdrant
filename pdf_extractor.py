from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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

