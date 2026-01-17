
import sqlite3
import os
from groq import Groq
import config
from utils.embeddings import EmbeddingEngine
from utils import prompts

class RAGAgent:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def get_documents(self, standard_ids):
        if not standard_ids:
            return []
            
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row
        placeholders = ','.join('?' * len(standard_ids))
        
        cursor = conn.execute(
            f"SELECT * FROM standards WHERE id IN ({placeholders})",
            standard_ids
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def process(self, query):
        # 1. Search relevant IDs
        top_ids = self.embedding_engine.search(query)
        
        # 2. Get full content
        docs = self.get_documents(top_ids)
        
        # 3. Format context
        context = "\n\n".join([
            f"Ref: ISO {d['id']}\nTitle: {d['title_en']}\nAbstract: {d['abstract']}" 
            for d in docs
        ])
        
        # 4. Generate answer
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompts.RAG_PROMPT.format(query=query, context=context)},
                {"role": "user", "content": query}
            ],
            model=config.GROQ_MODEL,
        )
        
        response = completion.choices[0].message.content
        
        # Clean up thinking blocks from Qwen3 (e.g., <think>...</think>)
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        return {
            "response": response,
            "source_documents": docs
        }
