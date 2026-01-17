
# ISO Standards Intelligence Assistant

This application uses a multi-agent RAG (Retrieval Augmented Generation) architecture to explore ISO Open Data.

## Setup

1. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   Run the data preparation script to download ISO data and generate embeddings/database.
   ```bash
   python prepare_data.py
   ```
   *This process may take a few minutes as it downloads files, creates a SQLite DB, and generates embeddings.*

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

## Configuration

- **Groq API Key**: You will need a Groq API Key to use the AI features. Enter it in the sidebar when the app launches.
- **Data Source**: ISO Open Data (active standards, filtered to top 3000 for relevance).

## Architecture

- **RAG Agent**: Semantically searches standard documents.
- **SQL Agent**: Queries structured metadata (counts, dates, etc.).
- **Viz Agent**: Generates Plotly charts.
- **Synthesis Agent**: Combines all insights into a final answer.
