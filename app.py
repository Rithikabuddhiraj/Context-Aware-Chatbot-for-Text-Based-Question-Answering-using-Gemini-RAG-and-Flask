from flask import Flask, render_template, request
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

app = Flask(__name__)

# Set your Google GenAI API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCJnR3hnj2m7XCMr_9OV5hg0PlwZ12ZTnc"

# Load and split text
loader = TextLoader("data/project_text_file.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Embedding and FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Gemini model setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=False
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    response = qa_chain.run(question)
    return render_template("result.html", question=question, response=response)

if __name__ == "__main__":
    app.run(debug=True)
