from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from PIL import Image
import pytesseract
import docx2txt



st.set_page_config(page_title="PDF Assistant", page_icon=":robot:")
st.title("DocQuest: Empowering Your Documents with AI - Answering and Summarizing with LLMs")
st.subheader('AI App Implemented By [Maximilien Kpizingui](https://kpizmax.hashnode.dev)')
'''
Say goodbye to manual PDF and DOCX and PNG files text searching and summarization! Our AI-powered app instantly extracts valuable insights from your PDF, DOCX and PNG file, saving you time and effort.With a user-friendly interface, upload your PDF, ask questions, and receive accurate answers in seconds.
Our intelligent algorithms analyze the text, understand context, and provide precise answers and summarization even for complex queries.Keep track of your queries with session history and easily clear it when needed.
Experience the power of AI to unlock information from PDFs,DOCXs and Image text files  with our secure and efficient AI App.
'''
st.image("post.jpg")

if 'session_history' not in st.session_state:
    st.session_state['session_history'] = []

def main():
    load_dotenv()

    st.sidebar.title("File Question Answering")
    st.sidebar.subheader("Menu")
    menu_options = ["Upload PDF", "Upload Word", "Upload Image", "Question Answering", "Session History"]
    choice = st.sidebar.selectbox("Select an option", menu_options)

    if choice == "Upload PDF":
        upload_pdf()
    elif choice == "Upload Word":
        upload_word()
    elif choice == "Upload Image":
        upload_image()
    elif choice == "Question Answering":
        perform_question_answering()
    elif choice == "Session History":
        display_session_history()

def upload_pdf():
    st.subheader("Upload your PDF document")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf is not None:
        st.session_state['uploaded_pdf'] = pdf
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        extracted_text = text
        st.subheader("PDF Extracted Text")
        st.text(extracted_text)

def upload_word():
    st.subheader("Upload your Word document")
    document = st.file_uploader("Choose a Word document", type=["docx"])

    if document is not None:
        st.session_state['uploaded_word'] = document
        text = docx2txt.process(document)

 
        extracted_text = text
        st.subheader("Docx Extracted Text")
        st.text(extracted_text)

def upload_image():
    st.subheader("Upload your image")
    image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if image is not None:
        st.session_state['uploaded_image'] = image
        img = Image.open(image)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Extract text from the uploaded image
        extracted_text = pytesseract.image_to_string(img)
        st.subheader("Image Extracted Text")
        st.text(extracted_text)

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
        elif 'uploaded_image' in st.session_state:
            img = Image.open(st.session_state['uploaded_image'])
            text = pytesseract.image_to_string(img)

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
