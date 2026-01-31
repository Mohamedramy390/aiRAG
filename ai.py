from flask import Flask, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient
from collections import Counter
from datetime import datetime
from bson import ObjectId

genai.configure(api_key=“AIzaSyB2gVLCb74-2z48TSzG1qm24SARPMOgWJs”)
model = genai.GenerativeModel(“gemini-2.5-pro”)

app = Flask(*name*)

# MongoDB connection

client = MongoClient(“mongodb://localhost:27017/”)
db = client[“education_db”]
topics_collection = db[“topics”]
questions_collection = db[“questions”]

@app.route(”/generate”, methods=[“POST”])
def generate():
data = request.json
query = data.get(“query”, “”)
context = data.get(“context”, None)


if not context:
    return jsonify({"error": "No context provided"}), 400

# Build prompt
prompt = f"""
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}
Answer:
"""

response = model.generate_content(prompt)
return jsonify({"answer": response.text})


@app.route(”/topic”, methods=[“POST”])
def add_topic():
“”“Add a new topic/headline”””
data = request.json
headline = data.get(“headline”, “”)


if not headline:
    return jsonify({"error": "No headline provided"}), 400

topic = {
    "headline": headline,
    "timestamp": datetime.now()
}

result = topics_collection.insert_one(topic)

return jsonify({
    "message": "Topic added successfully",
    "topic_id": str(result.inserted_id),
    "headline": headline
}), 201


@app.route(”/topics”, methods=[“GET”])
def get_topics():
“”“Get all topics”””
topics = list(topics_collection.find())


# Convert ObjectId to string for JSON serialization
for topic in topics:
    topic["_id"] = str(topic["_id"])
    topic["timestamp"] = topic["timestamp"].isoformat()

return jsonify({"topics": topics})


@app.route(”/question”, methods=[“POST”])
def add_question():
“”“Add a student question and connect it to a topic”””
data = request.json
question = data.get(“question”, “”)
topic_id = data.get(“topic_id”, None)


if not question:
    return jsonify({"error": "No question provided"}), 400

# Validate topic_id if provided
if topic_id:
    try:
        topic = topics_collection.find_one({"_id": ObjectId(topic_id)})
        if not topic:
            return jsonify({"error": "Invalid topic_id"}), 400
    except:
        return jsonify({"error": "Invalid topic_id format"}), 400

question_entry = {
    "question": question,
    "topic_id": topic_id,
    "timestamp": datetime.now()
}

result = questions_collection.insert_one(question_entry)

topic_headline = topic.get("headline", "No topic") if topic_id and topic else "No topic"

return jsonify({
    "message": "Question recorded successfully",
    "question_id": str(result.inserted_id),
    "question": question,
    "topic": topic_headline
}), 201


@app.route(”/questions”, methods=[“GET”])
def get_questions():
“”“Get all questions, optionally filtered by topic_id”””
topic_id = request.args.get(“topic_id”)


query = {}
if topic_id:
    query["topic_id"] = topic_id

questions = list(questions_collection.find(query))

# Convert ObjectId to string for JSON serialization
for question in questions:
    question["_id"] = str(question["_id"])
    question["timestamp"] = question["timestamp"].isoformat()

return jsonify({"questions": questions})


@app.route(”/analytics/repeated-questions”, methods=[“GET”])
def analyze_repeated_questions():
“”“Analyze the most repeated questions”””
topic_id = request.args.get(“topic_id”)


# Build query
query = {}
if topic_id:
    query["topic_id"] = topic_id

# Get questions from MongoDB
questions = list(questions_collection.find(query))

# Extract and normalize question text
question_texts = [q["question"].lower().strip() for q in questions]

# Count question frequency
question_counts = Counter(question_texts)

# Get top questions
top_questions = [
    {
        "question": question,
        "count": count,
        "percentage": round((count / len(question_texts) * 100), 2) if question_texts else 0
    }
    for question, count in question_counts.most_common(10)
]

# Get topic name if filtering by topic
topic_name = "All topics"
if topic_id:
    try:
        topic = topics_collection.find_one({"_id": ObjectId(topic_id)})
        if topic:
            topic_name = topic["headline"]
    except:
        pass

return jsonify({
    "total_questions": len(question_texts),
    "unique_questions": len(question_counts),
    "top_repeated_questions": top_questions,
    "topic": topic_name
})


@app.route(”/analytics/by-topic”, methods=[“GET”])
def analyze_by_topic():
“”“Get question statistics grouped by topic”””
topics = list(topics_collection.find())
topic_stats = []


for topic in topics:
    topic_id = str(topic["_id"])
    topic_questions = list(questions_collection.find({"topic_id": topic_id}))
    
    topic_stats.append({
        "topic_id": topic_id,
        "headline": topic["headline"],
        "total_questions": len(topic_questions),
        "sample_questions": [q["question"] for q in topic_questions[:5]]
    })

# Sort by most questions
topic_stats.sort(key=lambda x: x["total_questions"], reverse=True)

# Count questions without topic
no_topic_questions = questions_collection.count_documents({"topic_id": None})

return jsonify({
    "topics_with_stats": topic_stats,
    "total_topics": len(topics),
    "total_questions": questions_collection.count_documents({}),
    "questions_without_topic": no_topic_questions
})


@app.route(”/analytics/similarity”, methods=[“POST”])
def analyze_similar_questions():
“”“Use AI to find semantically similar questions”””
data = request.json
topic_id = data.get(“topic_id”)


# Build query
query = {}
if topic_id:
    query["topic_id"] = topic_id

questions = list(questions_collection.find(query))
question_texts = [q["question"] for q in questions]

if len(question_texts) < 2:
    return jsonify({"message": "Not enough questions to analyze"})

# Use AI to group similar questions
prompt = f"""
Analyze these student questions and group similar ones together.
Return the analysis as a structured response showing:
1. Common themes or topics
2. Groups of similar questions
3. Most frequently asked concepts

Questions:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(question_texts)])}
"""

response = model.generate_content(prompt)

return jsonify({
    "total_questions": len(question_texts),
    "similarity_analysis": response.text
})


if *name* == “*main*”:
app.run(host=“0.0.0.0”, port=5001)