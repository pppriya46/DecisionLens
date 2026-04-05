import os
from openai import OpenAI
from dotenv import load_dotenv
from api.search_service import search_similar_incidents
from api.telemetry import emit_latency, now_ms, elapsed_ms

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAG_SOURCE_COUNT = 3
MAX_CONTEXT_DESCRIPTION_CHARS = 160
MAX_CONTEXT_RESOLUTION_CHARS = 240


def truncate_text(value: str, limit: int) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."

def generate_rag_response(query_text: str, query_category: str = None, query_embedding=None) -> dict:
    print(f"\n[RAG] Query: {query_text[:80]}...")
    request_start_ms = now_ms()


    search_results = search_similar_incidents(
        query_text=query_text,
        query_category=query_category,
        query_embedding=query_embedding,
        top_k=20,
        top_n=RAG_SOURCE_COUNT
    )
    retrieval_duration_ms = search_results.get('timings_ms', {}).get('total')

    if not search_results['results']:
        total_duration_ms = elapsed_ms(request_start_ms)
        emit_latency(
            "rag_request",
            component="generate_rag_response",
            query_category=query_category,
            source_count=0,
            retrieval_duration_ms=retrieval_duration_ms,
            llm_duration_ms=0,
            embedding_source=search_results.get('embedding_source'),
            duration_ms=total_duration_ms,
        )
        return {
            "query": query_text,
            "answer": "I couldn't find any similar incidents in our database. Please provide more details or contact support directly.",
            "source_incidents": [],
            "confidence": "low",
            "timings_ms": {
                "retrieval": retrieval_duration_ms,
                "llm": 0,
                "total": total_duration_ms,
            },
            "embedding_source": search_results.get('embedding_source'),
        }

   
    context_parts = []
    for i, inc in enumerate(search_results['results'], 1):
        short_description = truncate_text(inc['description'], MAX_CONTEXT_DESCRIPTION_CHARS)
        short_resolution = truncate_text(inc['resolution'] or "", MAX_CONTEXT_RESOLUTION_CHARS)
        context_parts.append(
            f"Incident {i}\n"
            f"Issue: {short_description}\n"
            f"Type: {inc['issue_type'] or 'unknown'} | Product: {inc['product_area'] or 'unknown'}\n"
            f"Status: {inc['status'] or 'unknown'}\n"
            f"Resolution: {short_resolution or 'No resolution recorded.'}"
        )

    context = "\n---\n".join(context_parts)


    system_prompt = (
        "You are a support triage assistant.\n\n"
        "Your job is to produce short, practical troubleshooting guidance for a support analyst "
        "using only the retrieved historical incidents and their past resolutions.\n\n"
        "Rules:\n"
        "- Use only the provided incident context and source resolutions.\n"
        "- Do not invent fixes that are not supported by the sources.\n"
        "- If multiple sources suggest the same fix, summarize it once.\n"
        "- Do not repeat the same troubleshooting step in different words.\n"
        "- Keep the answer concise and operational.\n"
        "- If the sources are weak or mixed, say the guidance is tentative.\n\n"
        "Return exactly this format:\n"
        "Likely issue: <one sentence>\n\n"
        "Next steps:\n"
        "1. <one short sentence>\n"
        "2. <one short sentence>\n"
        "3. <one short sentence>\n\n"
        "Escalate if: <one sentence>"
    )

    user_prompt = f"""Current incident:
Description: {query_text}
Issue category: {query_category or 'unknown'}

Retrieved historical incidents:
{context}

Write short support guidance for the current incident using only the retrieved incidents."""

    print("[RAG] Calling GPT-4...")
    llm_start_ms = now_ms()
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=220
    )
    llm_duration_ms = elapsed_ms(llm_start_ms)
    emit_latency(
        "llm_response",
        component="rag_generation",
        model="gpt-4",
        source_count=len(search_results['results']),
        duration_ms=llm_duration_ms,
    )

    answer = completion.choices[0].message.content

    avg_similarity = sum(inc['scores']['similarity'] for inc in search_results['results']) / len(search_results['results'])
    
    if avg_similarity > 0.7:
        confidence = "high"
    elif avg_similarity > 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    total_duration_ms = elapsed_ms(request_start_ms)
    emit_latency(
        "rag_request",
        component="generate_rag_response",
        query_category=query_category,
        source_count=len(search_results['results']),
        retrieval_duration_ms=retrieval_duration_ms,
        llm_duration_ms=llm_duration_ms,
        embedding_source=search_results.get('embedding_source'),
        duration_ms=total_duration_ms,
    )

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
        "avg_similarity": round(avg_similarity, 3),
        "timings_ms": {
            "retrieval": retrieval_duration_ms,
            "llm": llm_duration_ms,
            "total": total_duration_ms,
        },
        "embedding_source": search_results.get('embedding_source'),
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
