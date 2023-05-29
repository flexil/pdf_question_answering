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



st.set_page_config(page_title="DocQuest", page_icon=":robot:")
st.title("DocQuest: Empowering Your Documents with AI - Answering and Summarizing with LLMs")
st.subheader('AI App Implemented By [Maximilien Kpizingui](https://kpizmax.hashnode.dev)')
'''
This app has been deleted by the creator
'''
st.image("post.jpg")

