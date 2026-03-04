"""
Flask API Entry Point - DecisionLens
Exposes RAG and Search endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)


# ── Health Check ─────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status":  "healthy",
        "service": "DecisionLens API",
        "version": "1.0.0"
    })


# ── RAG Endpoint ─────────────────────────────────────────────────────────
@app.route('/api/rag', methods=['POST'])
def rag_endpoint():
    """
    Full RAG pipeline endpoint
    Request : { "query": "...", "category": "Software", "top_k": 5 }
    Response: { similar_incidents, resolution_guidance, source, cached }
    """
    try:
        from api.rag_service import rag_query

        data          = request.get_json()
        query_text    = data.get('query')
        query_category = data.get('category', None)
        top_k         = data.get('top_k', 5)

        if not query_text:
            return jsonify({"error": "query field is required"}), 400

        result = rag_query(query_text, query_category, top_k)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Search Endpoint ───────────────────────────────────────────────────────
@app.route('/api/search', methods=['POST'])
def search_endpoint():
    """
    Similarity search only (no GPT-4)
    Request : { "query": "...", "category": "Software", "top_k": 5 }
    Response: { total_candidates, results }
    """
    try:
        from api.search_service import search_similar_incidents

        data           = request.get_json()
        query_text     = data.get('query')
        query_category = data.get('category', None)
        top_k          = data.get('top_k', 5)

        if not query_text:
            return jsonify({"error": "query field is required"}), 400

        result = search_similar_incidents(query_text, query_category, top_k=20, top_n=top_k)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=True
    )