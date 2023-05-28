from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Initialize session state

st.set_page_config(page_title="PDF Assistant", page_icon=":robot:")
st.title("PDF Question Answering App Powered by LLMs")
st.subheader('AI Web App by [Maximilien Kpizingui](https://kpizmax.hashnode.dev)')
'''
Say goodbye to manual PDF searching! Our AI-powered app instantly extracts valuable insights from your PDF files, saving you time and effort.With a user-friendly interface, upload your PDF, ask questions, and receive accurate answers in seconds.
Our intelligent algorithms analyze the text, understand context, and provide precise answers, even for complex queries.Keep track of your queries with session history and easily clear it when needed.
Experience the power of AI to unlock information from PDFs with our secure and efficient PDF Question Answering App.
'''
st.image("post.jpg")
st.sidebar.markdown("Email: maximilien@tutanota.de")
st.sidebar.markdown("Element ID: @maximilien:matrix.org")
#Initialize session state
if 'session_history' not in st.session_state:
    st.session_state['session_history'] = []


def main():
    load_dotenv()
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text and create embeddings when PDF is uploaded
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Display the extracted text
        st.subheader("PDF Extracted Content")
        st.text(text)
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
      
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
      
        # show user input and search button
        user_question = st.text_input("Ask a question about your PDF:")
        if st.button("Search"):
            # Add the question and answer to the session history
            st.session_state['session_history'].append((user_question, ""))
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
                
            # Update the latest answer in the session history
            st.session_state['session_history'][-1] = (user_question, response)
            
            st.write(response)
        
        # Display session history
        st.subheader("Session History")
        for question, answer in st.session_state['session_history']:
            st.write("Question:", question)
            st.write("Answer:", answer)
            st.write("----")
        
        if st.button("Clear"):
            # Clear the session history
            st.session_state['session_history'] = []

if __name__ == '__main__':
    main()
