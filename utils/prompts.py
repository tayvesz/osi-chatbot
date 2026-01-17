
RAG_PROMPT = """You are an expert in ISO international standards.
Find relevant standards for: {query}

Context (standards metadata):
{context}

Provide:
- Standard reference (ISO XXXXX:YYYY)
- Title
- Brief summary
- Relevance to query
"""

SQL_PROMPT = """You are a SQL expert for ISO standards database.

Schema:
- standards: id, title_en, title_fr, abstract, publicationDate, status, icsCode, ownerCommittee, year, full_text
- committees: id, title_en, title (raw name)

User question: {query}

Generate a SQL query to answer this question.
Return only the SQL query, no explanation.
Do not use markdown formatting like ```sql. Just the raw query.
"""

SYNTHESIS_PROMPT = """You are an ISO standards expert assistant.

User question: {query}

Available information:
1. Relevant standards (from RAG):
{rag_results}

2. Statistical analysis (from SQL):
{sql_results}

3. Visual insights (from charts):
{viz_description}

Provide a comprehensive answer.
IMPORTANT: The User Interface DOES display the charts mentioned in "Visual insights" right below your response.
- If {viz_description} indicates a chart was generated, mention it (e.g., "As shown in the chart below...").
- DO NOT say "I cannot generate charts". You are part of a system that HAS generated one.
- List relevant ISO standard references
- Explain how they address the query
- Highlight key statistics from the SQL data
- Suggest related standards if applicable

Keep it concise and actionable.
"""
