import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize the Ollama model
ollama_model = Ollama(model="mistral")


# Define a function to process input and generate answers
def process_input(context, question):
    # Define RAG pipeline
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
            {"context": context, "question": question}
            | after_rag_prompt
            | ollama_model
            | RunnablePassthrough()  # No need for further processing, just pass through
            | StrOutputParser()  # Convert the output to a string
    )
    return after_rag_chain.invoke()


# Streamlit app
def main():
    st.title("Question Answering with Ollama and RAG")
    context_option = st.radio("Select context option:", ("Paste the context", "Upload a document"))

    if context_option == "Paste the context":
        context = st.text_area("Paste the context here:")
    else:
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
        if uploaded_file is not None:
            context = uploaded_file.getvalue().decode("latin-1")  # Use 'latin-1' encoding
        else:
            context = None

    question = st.text_input("Ask your question:")

    if st.button("Get Answer"):
        if context and question:
            answer = process_input(context, question)
            st.write("Answer:", answer)
        else:
            st.error("Please provide both context and question.")


if __name__ == "__main__":
    main()
