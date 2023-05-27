# PDF Question Answering

PDF Question Answering is an AI-powered application that allows users to extract answers from PDF files by asking questions. It leverages advanced natural language processing techniques and powerful libraries to provide quick and accurate responses.

## Features

- Upload PDF files: Users can easily upload their PDF files using the user-friendly interface.
- Extract text: The application extracts the text content from the uploaded PDF files using PyPDF2 library.
- Text processing and embeddings: The text is processed and split into chunks using langchain's CharacterTextSplitter, and semantic embeddings are generated using OpenAIEmbeddings.
- Building the knowledge base: The text chunks and embeddings are stored in FAISS, enabling efficient similarity search.
- User interaction and question answering: Users can input their questions, and the application utilizes the question answering chain and knowledge base to provide answers.
- Session history and clearing: The application keeps track of user queries and their answers, and the session history can be cleared with the "Clear" button.

## Installation

1. Clone the repository:

git clone https://github.com/flexil/pdf-question-answering.git


2. Install the required dependencies:

Certainly! Here's an example of a README file for your project:

vbnet

# PDF Question Answering

PDF Question Answering is an AI-powered application that allows users to extract answers from PDF files by asking questions. It leverages advanced natural language processing techniques and powerful libraries to provide quick and accurate responses.

## Features

- Upload PDF files: Users can easily upload their PDF files using the user-friendly interface.
- Extract text: The application extracts the text content from the uploaded PDF files using PyPDF2 library.
- Text processing and embeddings: The text is processed and split into chunks using langchain's CharacterTextSplitter, and semantic embeddings are generated using OpenAIEmbeddings.
- Building the knowledge base: The text chunks and embeddings are stored in FAISS, enabling efficient similarity search.
- User interaction and question answering: Users can input their questions, and the application utilizes the question answering chain and knowledge base to provide answers.
- Session history and clearing: The application keeps track of user queries and their answers, and the session history can be cleared with the "Clear" button.

## Installation

1. Clone the repository:

git clone https://github.com/your-username/pdf-question-answering.git

markdown


2. Install the required dependencies:

pip install -r requirements.txt


3. Set up the environment variables:

Create a `.env` file in the project root directory and add the necessary environment variables. Refer to the `.env.example` file for the required variables.

## Usage

1. Run the application:

streamlit run main.py


2. Upload your PDF file using the provided interface.
3. Ask a question about the PDF content in the text input field.
4. The application will display the answer based on the uploaded PDF file.
5. The session history will be shown above the search bar, and you can clear it by clicking the "Clear" button.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please create an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
