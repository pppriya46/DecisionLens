"""
RAG Integration with GPT-4 - Step 7
1. Build context from top-5 similar incidents (uses Step 6 search_service.py)
2. Prompt engineering: System message + context + task
3. Call GPT-4 API with retrieved incidents
4. Error handling: if GPT-4 fails, return ML + RAG results without LLM
5. Cache LLM responses (if exact same/similar incidents asked before)
"""

import os
import json
import hashlib
import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv
from api.search_service import search_similar_incidents

load_dotenv()

client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = "gpt-4"


_response_cache: dict = {}


def _generate_cache_key(similar_incidents: list) -> str:
    """
    Generate a unique cache key based on incident numbers.
    If exact same incidents are retrieved, we return cached response
    instead of calling GPT-4 again.

    Example:
        incidents = [INC001, INC002, INC003, INC004, INC005]
        cache_key = md5("INC001-INC002-INC003-INC004-INC005")
    """
    incident_numbers = "-".join([inc["number"] for inc in similar_incidents])
    return hashlib.md5(incident_numbers.encode()).hexdigest()


def build_context(similar_incidents: list) -> str:
    """
    Convert top-5 similar incidents into a formatted context string.
    This is what GPT-4 reads to understand historical resolutions.
    Richer context = better GPT-4 response.
    """
    if not similar_incidents:
        return "No similar historical incidents found."

    context_parts = [
        "Here are the top 5 most similar historical incidents:\n",
        "=" * 50
    ]

    for idx, inc in enumerate(similar_incidents, 1):
        context_parts.append(
            f"\nIncident {idx}: {inc['number']}\n"
            f"  State      : {inc['incident_state']}\n"
            f"  Category   : {inc['category']} > {inc['subcategory']}\n"
            f"  Symptom    : {inc['symptom']}\n"
            f"  Priority   : {inc['priority']}\n"
            f"  Impact     : {inc['impact']}\n"
            f"  Urgency    : {inc['urgency']}\n"
            f"  Resolution : {inc['resolution']}\n"
            f"  Similarity : {inc['scores']['similarity']:.1%}\n"
            f"  Resolved At: {inc['resolved_at'] or 'N/A'}\n"
        )

    context_parts.append("=" * 50)
    return "\n".join(context_parts)



def build_prompt(query_text: str, context: str) -> list:
    """
    Build the GPT-4 prompt using:
    - System message : defines GPT-4's role and behavior
    - User message   : context from similar incidents + the actual task

    Good prompt = specific role + clear context + clear task
    """

    system_message = """You are an expert IT incident management assistant for DecisionLens.
Your job is to analyze historical incident data and provide clear, actionable 
resolution guidance for new incidents.

Guidelines:
- Always base your recommendations on the historical incidents provided
- Be specific and actionable in your resolution steps
- Cite the incident numbers you are referencing (e.g. INC0012345)
- If incidents show different resolutions, explain which is most applicable and why
- Keep response concise but thorough (max 300 words)
- If no similar incidents are relevant, say so clearly"""

    user_message = f"""New Incident Requiring Resolution Guidance:
{query_text}

Historical Similar Incidents for Reference:
{context}

Based on these similar historical incidents, please provide:
1. Most likely root cause
2. Recommended resolution steps
3. Which historical incident is most relevant and why
4. Estimated priority if not already assigned"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message}
    ]


def call_gpt4(messages: list) -> str:
    """
    Call GPT-4 API with the engineered prompt.
    Returns GPT-4's resolution guidance as a string.
    Raises exception if API call fails (handled by caller).
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.3,   
        max_tokens=500
    )
    return response.choices[0].message.content


def build_fallback_response(similar_incidents: list) -> str:
    """
    If GPT-4 fails (API error, rate limit, etc.),
    return structured ML + RAG results WITHOUT the LLM.
    This ensures the system always returns something useful.
    """
    if not similar_incidents:
        return "No similar incidents found. Please escalate to senior support."


    resolutions = [
        inc["resolution"] for inc in similar_incidents
        if inc["resolution"]
    ]

    most_common = max(set(resolutions), key=resolutions.count) if resolutions else "Unknown"

    lines = [
        "[GPT-4 Unavailable - Showing RAG Results Only]\n",
        f"Most common resolution from similar incidents: {most_common}\n",
        "Top similar historical incidents:\n"
    ]

    for inc in similar_incidents:
        lines.append(
            f"  • {inc['number']} ({inc['scores']['similarity']:.1%} match)\n"
            f"    Category  : {inc['category']} > {inc['subcategory']}\n"
            f"    Resolution: {inc['resolution']}\n"
        )

    lines.append(
        "\nRecommendation: Follow the resolution pattern from the "
        "highest similarity incident above."
    )

    return "\n".join(lines)

"RAG Pipeline"
def rag_query(query_text: str, query_category: str = None, top_k: int = 5) -> dict:
    """
    Full RAG Pipeline - Step 7:

    1. Call Step 6 (search_service) to get top-5 similar incidents
    2. Build context string from those incidents
    3. Check cache - if same incidents seen before, return cached response
    4. Engineer prompt (system + context + task)
    5. Call GPT-4 for resolution guidance
    6. Cache the response for future identical queries
    7. Return full result dict

    If GPT-4 fails at any point → fallback to RAG results without LLM

    Args:
        query_text     : new incident description
        query_category : optional category hint for better search
        top_k          : number of similar incidents to retrieve (default 5)
    """
    print(f"\n{'='*60}")
    print(f"RAG Query: {query_text[:80]}...")
    print(f"{'='*60}")
    print("\n[1/5] Searching for similar incidents...")
    try:
        search_result = search_similar_incidents(
            query_text=query_text,
            query_category=query_category,
            top_k=20,   
            top_n=top_k 
        )
        similar_incidents = search_result["results"]
        print(f"      Found {len(similar_incidents)} similar incidents")

    except Exception as e:
        print(f"      Search failed: {e}")
        return {
            "query":              query_text,
            "similar_incidents":  [],
            "resolution_guidance": f"Search service unavailable: {str(e)}",
            "source":             "error",
            "cached":             False,
        }

    print("\n[2/5] Building context from similar incidents...")
    context = build_context(similar_incidents)

    print("\n[3/5] Checking response cache...")
    cache_key = _generate_cache_key(similar_incidents)

    if cache_key in _response_cache:
        print("      Cache HIT! Returning cached response")
        return {
            "query":               query_text,
            "similar_incidents":   similar_incidents,
            "resolution_guidance": _response_cache[cache_key],
            "source":              "gpt-4",
            "cached":              True,  
        }

    print("      Cache MISS - will call GPT-4")


    print("\n[4/5] Building GPT-4 prompt...")
    messages = build_prompt(query_text, context)


    print("\n[5/5] Calling GPT-4...")
    try:
        resolution_guidance = call_gpt4(messages)
        source = "gpt-4"
        print("      GPT-4 response received!")

        _response_cache[cache_key] = resolution_guidance
        print(f"      Response cached (cache size: {len(_response_cache)})")

    except Exception as e:
        print(f"      GPT-4 failed: {e}")
        print("      Using fallback RAG response without LLM...")
        resolution_guidance = build_fallback_response(similar_incidents)
        source = "fallback"

    return {
        "query":               query_text,
        "similar_incidents":   similar_incidents,
        "resolution_guidance": resolution_guidance,
        "source":              source,   # "gpt-4" or "fallback"
        "cached":              False,
    }


#Mannual TEST

if __name__ == "__main__":
    print("Testing RAG Pipeline - Step 7\n")

    
    result = rag_query(
        query_text="User cannot login to Outlook, password reset not working. "
                   "Affecting entire sales department since this morning.",
        query_category="Software",
        top_k=5
    )

    print("\n" + "="*60)
    print("FINAL RAG RESULT")
    print("="*60)
    print(f"Source  : {result['source']}")
    print(f"Cached  : {result['cached']}")
    print(f"\nSimilar Incidents:")
    for inc in result['similar_incidents']:
        print(f"  • {inc['number']} - {inc['scores']['similarity']:.1%} match"
              f" - Resolution: {inc['resolution']}")

    print(f"\nGPT-4 Resolution Guidance:")
    print(result['resolution_guidance'])
    print("="*60)

#Test 2 - Cache

    print("\n\nTesting cache - running same query again...")
    result2 = rag_query(
        query_text="User cannot login to Outlook, password reset not working.",
        query_category="Software",
        top_k=5
    )
    print(f"Cached: {result2['cached']}")  # Should print True