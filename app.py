from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import docx2txt

st.set_page_config(page_title="PDF Assistant", page_icon=":robot:")
st.title("PDF & DOCX Question Answering App Powered by LLMs")
st.subheader('AI Web App by [Maximilien Kpizingui](https://kpizmax.hashnode.dev)')
'''
Say goodbye to manual PDF searching! Our AI-powered app instantly extracts valuable insights from your PDF files, saving you time and effort.With a user-friendly interface, upload your PDF, ask questions, and receive accurate answers in seconds.
Our intelligent algorithms analyze the text, understand context, and provide precise answers, even for complex queries.Keep track of your queries with session history and easily clear it when needed.
Experience the power of AI to unlock information from PDFs with our secure and efficient PDF Question Answering App.
'''
st.image("post.jpg")

if 'session_history' not in st.session_state:
    st.session_state['session_history'] = []

def main():
    load_dotenv()

    
    st.sidebar.title("Question Answering")
    st.sidebar.subheader("Menu")
    menu_options = ["Upload PDF", "Upload Word", "Question Answering", "Session History"]
    choice = st.sidebar.selectbox("Select an option", menu_options)

    if choice == "Upload PDF":
        upload_pdf()
    elif choice == "Upload Word":
        upload_word()
    elif choice == "Question Answering":
        perform_question_answering()
    elif choice == "Session History":
        display_session_history()

def upload_pdf():
 #   st.title("Upload PDF")
    st.subheader("Upload your PDF document")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf is not None:
        st.session_state['uploaded_pdf'] = pdf
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.subheader("PDF Extracted Content")
        st.text(text)

def upload_word():
#    st.title("Upload Word")
    st.subheader("Upload your Word document")
    document = st.file_uploader("Choose a Word document", type=["docx"])

    if document is not None:
        st.session_state['uploaded_word'] = document
        text =docx2txt.process(document)

        st.subheader("Document Extracted Content")
        st.text(text)

def perform_question_answering():
    st.title("Question Answering")
    user_question = st.text_input("Ask a question:")
    if st.button("Search"):
        st.session_state['session_history'].append((user_question, ""))

        
        text = ""
        if 'uploaded_pdf' in st.session_state:
            pdf_reader = PdfReader(st.session_state['uploaded_pdf'])
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif 'uploaded_word' in st.session_state:
            text = docx2txt.process(st.session_state['uploaded_word'])

       
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

       
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)

        
        st.session_state['session_history'][-1] = (user_question, response)

       
        st.write(response)

def display_session_history():
    st.title("Session History")
    if len(st.session_state['session_history']) > 0:
        for question, answer in st.session_state['session_history']:
            st.write("Question:", question)
            st.write("Answer:", answer)
            st.write("----")
    else:
        st.write("No session history available.")

if __name__ == '__main__':
    main()
