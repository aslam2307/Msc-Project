from flask import Flask, render_template, request, jsonify
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Make sure this is properly initialized

app = Flask(__name__)

# Initialize LLM
model = OllamaLLM(model="llama3.2")

# Prompt template
template = """
You are an assistant for answering questions about the University of Roehampton.
Use ONLY the provided documents (course details and general info).
If the answer is not found, reply strictly: "I don’t have that information in my database."

Documents:
{docs}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"answer": "Please enter a question."})

    # Retrieve relevant documents (pass the question as a string!)
    retrieved_docs = retriever.invoke(question)
    docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Run the chain with docs and question
    response = chain.invoke({"docs": docs_text, "question": question})

    answer = response if isinstance(response, str) else response.get("result", "")

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
