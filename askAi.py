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

    prompt =f"""
    You are a strictly academic teaching assistant.
    Your goal is to answer the student's question based on the provided Context (lecture notes).

    Follow this logic strictly in order:
    1. **Direct Answer:** If the answer is explicitly found in the Context below, answer it using ONLY the context.
    2. **Definition:** If the user asks for a definition of a term that is MENTIONED in the Context (e.g., as a title, header, or bullet point) but not defined there, use your general knowledge to provide a standard academic definition.
    3. **Related but Missing:** If the question is NOT in the context but is clearly related to the **same subject matter or domain as the Context** (e.g., if the doc is about 'Roman History' and the question asks about 'Julius Caesar', or if the doc is about 'Biology' and the question asks about 'Cells'), answer it using your general knowledge, but START your answer with: "This is not explicitly in the lecture, but generally speaking..."    
    4. **Unrelated:** If the question is completely unrelated to the domain of the Context (e.g., asking about cooking when the doc is about physics), say: "This question is out of the scope of this lecture."

    Context:
    {context_snippet}

    Question: {query}
    Answer:
    """

    # ✅ Create a fresh model per request
    model = genai.GenerativeModel("gemini-2.5-flash")
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
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    
    topics = [t.strip() for t in response.text.strip().split("\n") if t.strip()]
    return jsonify({"topics": topics})


@app.route("/categorize", methods=["POST"])
def categorize_question():
    """
    Determines which topic is most related to a given question.
    """
    data = request.json
    #print("Received categorize request:", data)
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
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    related_topic = response.text.strip()
    
    return jsonify({"related_topic": related_topic})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
