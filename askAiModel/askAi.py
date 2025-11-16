from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
# ⚙️ Load API key safely from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    """
    Answers a question based on the provided context.
    """
    data = request.json
    query = data.get("query", "")
    context = data.get("context")

    if not context:
        return jsonify({"error": "No context provided"}), 400

    # Limit input size to keep prompt short
    context_snippet = context[:8000]

    prompt = f"""
    Use ONLY the following context to answer the question.
    If the answer is not in the context, say "I don't know."

    Context:
    {context_snippet}

    Question: {query}
    Answer:
    """

    # ✅ Create a fresh model per request
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return jsonify({"answer": response.text.strip()})


@app.route("/topics", methods=["POST"])
def generate_topics():
    """
    Extracts main topic titles from educational or technical text.
    """
    data = request.json
    context = data.get("context")

    if not context:
        return jsonify({"error": "No context provided"}), 400

    # Use only the first part of the context (for efficiency)
    context_snippet = context[:6000]

    prompt = f"""
    Analyze the following text and extract its main sections or topic titles.
    Return ONLY a newline-separated list of topic titles (no numbering or explanations).

    Text:
    {context_snippet}

    Topics:
    """

    # ✅ Fresh model to avoid memory context
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    
    topics = [t.strip() for t in response.text.strip().split("\n") if t.strip()]
    return jsonify({"topics": topics})


@app.route("/categorize", methods=["POST"])
def categorize_question():
    """
    Determines which topic is most related to a given question.
    """
    data = request.json
    query = data.get("query")
    topics = data.get("topics")

    if not query or not topics:
        return jsonify({"error": "Both 'query' and 'topics' must be provided"}), 400
    
    formatted_topics = "\n".join(f"- {topic}" for topic in topics)

    prompt = f"""
    Given the following list of topics, identify which single topic is most relevant to the question.
    Return ONLY the exact topic title from the list.

    Topics:
    {formatted_topics}

    Question: {query}

    Most Relevant Topic:
    """

    # ✅ Fresh model per request
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    related_topic = response.text.strip()
    
    return jsonify({"related_topic": related_topic})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
