import os
from openai import OpenAI
from dotenv import load_dotenv
from api.search_service import search_similar_incidents

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_rag_response(query_text: str, query_category: str = None) -> dict:
    print(f"\n[RAG] Query: {query_text[:80]}...")


    search_results = search_similar_incidents(
        query_text=query_text,
        query_category=query_category,
        top_k=20,
        top_n=5
    )

    if not search_results['results']:
        return {
            "query": query_text,
            "answer": "I couldn't find any similar incidents in our database. Please provide more details or contact support directly.",
            "source_incidents": [],
            "confidence": "low"
        }

   
    context_parts = []
    for i, inc in enumerate(search_results['results'], 1):
        context_parts.append(
            f"Incident {i}:\n"
            f"Issue: {inc['description']}\n"
            f"Type: {inc['issue_type']} | Product: {inc['product_area']}\n"
            f"Priority: {inc['priority']} | Status: {inc['status']}\n"
            f"Resolution: {inc['resolution']}\n"
            f"Customer Sentiment: {inc['sentiment']} | CSAT: {inc['csat_score']}\n"
        )

    context = "\n---\n".join(context_parts)


    system_prompt = """You are an IT support assistant. Based on similar past incidents, provide helpful troubleshooting advice.

Instructions:
1. Analyze the similar incidents provided
2. Give step-by-step troubleshooting advice
3. Mention if escalation might be needed
4. Be concise and actionable
5. If multiple incidents show similar resolutions, highlight that pattern"""

    user_prompt = f"""User Query: {query_text}

Similar Past Incidents:
{context}

Based on these similar incidents, what troubleshooting steps or solution would you recommend?"""

    print("[RAG] Calling GPT-4...")
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    answer = completion.choices[0].message.content

    avg_similarity = sum(inc['scores']['similarity'] for inc in search_results['results']) / len(search_results['results'])
    
    if avg_similarity > 0.7:
        confidence = "high"
    elif avg_similarity > 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "query": query_text,
        "answer": answer,
        "source_incidents": [
            {
                "ticket_id": inc['ticket_id'],
                "issue_type": inc['issue_type'],
                "product_area": inc['product_area'],
                "description": inc['description'][:100] + "..." if len(inc['description']) > 100 else inc['description'],
                "resolution": inc['resolution'][:100] + "..." if len(inc['resolution'] or "") > 100 else inc['resolution'],
                "similarity_score": inc['scores']['similarity']
            }
            for inc in search_results['results']
        ],
        "confidence": confidence,
        "avg_similarity": round(avg_similarity, 3)
    }


if __name__ == "__main__":
    result = generate_rag_response(
        query_text="I can't login to my account, password reset not working",
        query_category="account_access"
    )
    
    print("\n" + "="*60)
    print("RAG RESPONSE")
    print("="*60)
    print(f"Query: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']}")
    print(f"Based on {len(result['source_incidents'])} similar incidents")