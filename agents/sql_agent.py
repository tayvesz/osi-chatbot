
import sqlite3
import pandas as pd
import os
from groq import Groq
import config
from utils import prompts

class SQLAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def execute_query(self, sql_query):
        try:
            conn = sqlite3.connect(config.DB_PATH)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
        except Exception as e:
            return f"Error executing query: {e}"

    def process(self, query):
        # 1. Generate SQL
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompts.SQL_PROMPT.format(query=query)},
                {"role": "user", "content": query}
            ],
            model=config.GROQ_MODEL,
        )
        
        generated_sql = completion.choices[0].message.content.strip()
        
        # Clean up potential thinking blocks from Qwen3 (e.g., <think>...</think>)
        import re
        generated_sql = re.sub(r'<think>.*?</think>', '', generated_sql, flags=re.DOTALL).strip()
        
        # Clean up potential markdown code blocks
        if "```sql" in generated_sql:
            generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in generated_sql:
            parts = generated_sql.split("```")
            if len(parts) >= 2:
                generated_sql = parts[1].strip()
        
        # Remove any remaining HTML-like tags
        generated_sql = re.sub(r'<[^>]+>', '', generated_sql).strip()
        
        # Remove any leading/trailing quotes that some models add
        generated_sql = generated_sql.strip('"').strip("'")
            
        print(f"Generated SQL: {generated_sql}")
        
        # 2. Execute SQL
        results = self.execute_query(generated_sql)
        
        return {
            "query": generated_sql,
            "results": results
        }
