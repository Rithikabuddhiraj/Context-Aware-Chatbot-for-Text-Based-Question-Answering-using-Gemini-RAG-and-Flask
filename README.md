# Context-Aware Chatbot for Text-Based Question Answering using Gemini RAG and Flask

This project implements a lightweight, context-aware chatbot capable of answering questions from custom text documents using the **Retrieval-Augmented Generation (RAG)** approach. It combines **LangChain**, **Google's Gemini 1.5 Flash** via API, and **Flask** to create a user-friendly, local web interface.

---

## 🔍 Project Objective

To build a web-based intelligent chatbot that:
- Reads domain-specific or project-specific `.txt` files.
- Retrieves relevant context chunks using **FAISS vector store** and **HuggingFace embeddings**.
- Generates accurate answers using the **Gemini LLM (via Google Generative AI API)**.

---

## 🚀 Tech Stack

- **Frontend:** HTML + Flask templates  
- **Backend:** Python, Flask  
- **LLM:** Gemini 1.5 Flash (via `langchain-google-genai`)  
- **Vector Store:** FAISS (for semantic search)  
- **Embeddings:** HuggingFace MiniLM (`all-MiniLM-L6-v2`)  
- **Frameworks:** LangChain, Flask

---

## 🧠 How It Works

1. Loads a custom text document (`.txt`) using LangChain’s `TextLoader`.
2. Splits the text into meaningful overlapping chunks for better semantic understanding.
3. Converts chunks into embeddings using HuggingFace.
4. Stores embeddings in a FAISS vector database.
5. On user query, retrieves relevant chunks and passes them to Gemini via LangChain.
6. Displays the generated response on a Flask-powered web UI.

---

## 📂 Folder Structure

```

├── app.py                   # Flask backend logic
├── data/
│   └── project\_text\_file.txt  # Input text data
├── templates/
│   ├── index.html            # Input page
│   └── result.html           # Response page
├── static/                  # (Optional CSS folder)
│   └── styles.css
└── README.md

````

---

## ⚙️ Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Rithikabuddhiraj/Context-Aware-Chatbot-for-Text-Based-Question-Answering-using-Gemini-RAG-and-Flask.git
   cd Context-Aware-Chatbot-...
````

2. **Install required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Gemini API Key**
   Replace the placeholder inside `app.py`:

   ```python
   os.environ["GOOGLE_API_KEY"] = "your_google_api_key"
   ```

4. **Run the Flask app**

   ```bash
   python app.py
   ```

5. **Open in browser:**
   Visit `http://127.0.0.1:5000`

---

## 📒 Example Usage

* Upload a `.txt` file containing your domain knowledge (e.g., product FAQs, policy notes, research summary).
* Ask questions like:

  * *"What is the main idea of this project?"*
  * *"Explain the deployment setup."*
* The chatbot will fetch relevant content and respond intelligently.

---

## 📎 Notebook Reference

You can also view the logic and RAG implementation in the Colab notebook attached above.


