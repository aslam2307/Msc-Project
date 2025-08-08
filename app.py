from flask import Flask, request, jsonify, render_template
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = Flask(__name__)

model = OllamaLLM(model="llama3.2")

template = """
You are an expert on the University of Roehampton in the UK.

Use only the information provided in the context below to answer the user's question. 
If the answer cannot be found in the context, say "I don't know based on the data provided."

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    answer = chain.invoke({"context": context, "question": question})
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
