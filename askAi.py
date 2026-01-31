from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ⚙️ Configuration
# Make sure HUGGINGFACE_API_KEY is in your .env file
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Requires accepted license on HF
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" # Open access model, good fallback

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

def get_llama_response(messages, temperature=0.3):
    """Helper function to call Llama 3.1 cleanly"""
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=2048, # increased for longer answers if needed
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # This usually catches 401 (Bad Token) or 403 (License not accepted yet)
        print(f"Error calling Hugging Face: {e}")
        return None

@app.route("/generate", methods=["POST"])
def generate():
    """
    Answers a question based on the provided context.
    """
    data = request.json
    query = data.get("query", "")
    context = data.get("context", "")

    if not context:
        return jsonify({"error": "No context provided"}), 400

    # Llama 3.1 Context Limit is 128k, so 8000 chars is very safe
    context_snippet = context[:15000] 

    # We move the rigorous logic into the SYSTEM prompt for better adherence
    system_instruction = """
    You are a strictly academic teaching assistant.
    Your goal is to answer the student's question based on the provided Context (lecture notes).

    Follow this logic strictly in order:
    1. **Direct Answer:** If the answer is explicitly found in the Context below, answer it using ONLY the context.
    2. **Definition:** If the user asks for a definition of a term MENTIONED in the Context but not defined there, use general knowledge.
    3. **Related but Missing:** If the question is NOT in the context but is clearly related to the domain, answer it starting with: "This is not explicitly in the lecture, but generally speaking..."
    4. **Unrelated:** If the question is completely unrelated, say: "This question is out of the scope of this lecture."
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Context:\n{context_snippet}\n\nQuestion: {query}"}
    ]

    answer = get_llama_response(messages)
    
    if answer is None:
        return jsonify({"error": "Failed to generate response. Check API Key or License approval."}), 500

    return jsonify({"answer": answer})


@app.route("/topics", methods=["POST"])
def generate_topics():
    """
    Extracts main topic titles.
    """
    data = request.json
    context = data.get("context", "")

    if not context:
        return jsonify({"error": "No context provided"}), 400

    context_snippet = context[:10000]

    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"""
        Analyze the following text and extract its main sections or topic titles.
        Return ONLY a newline-separated list of topic titles (no numbering, no bullets, no explanations).

        Text:
        {context_snippet}
        """}
    ]

    # Lower temperature for extraction tasks to be more deterministic
    topics_text = get_llama_response(messages, temperature=0.1)
    
    if topics_text is None:
        return jsonify({"error": "API Error"}), 500

    # Clean up the list
    topics = [t.strip().replace("* ", "").replace("- ", "") for t in topics_text.split("\n") if t.strip()]
    
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

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
        Given the following list of topics, identify which single topic is most relevant to the question.
        Return ONLY the exact topic title from the list.

        Topics:
        {formatted_topics}

        Question: {query}
        """}
    ]

    related_topic = get_llama_response(messages, temperature=0.1)

    if related_topic is None:
        return jsonify({"error": "API Error"}), 500
    
    return jsonify({"related_topic": related_topic})


if __name__ == "__main__":
    # Get the PORT from Render (default to 5001 only if running locally)
    port = int(os.environ.get("PORT", 5001))
    
    # Listen on 0.0.0.0 (Required for Render)
    app.run(host="0.0.0.0", port=port)