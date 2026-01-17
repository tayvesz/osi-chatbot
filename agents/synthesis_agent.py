
import os
import pandas as pd
from groq import Groq
import config
from utils import prompts

class SynthesisAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def process(self, query, rag_response, sql_response, viz_type=None):
        # Format RAG results
        rag_text = rag_response.get("response", "No documents found.")
        
        # Format SQL results
        sql_data = sql_response.get("results")
        if isinstance(sql_data, pd.DataFrame):
            if sql_data.empty:
                sql_text = "No statistical data found."
            else:
                sql_text = sql_data.to_markdown(index=False)
        else:
            sql_text = str(sql_data)
            
        viz_text = f"An interactive {viz_type} chart was generated." if viz_type else "No visualization generated."
        
        # Generate final answer
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompts.SYNTHESIS_PROMPT.format(
                    query=query,
                    rag_results=rag_text,
                    sql_results=sql_text,
                    viz_description=viz_text
                )},
                {"role": "user", "content": query}
            ],
            model=config.GROQ_MODEL,
        )
        
        response = completion.choices[0].message.content
        
        # Clean up thinking blocks from Qwen3 (e.g., <think>...</think>)
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        return response
