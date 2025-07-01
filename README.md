# RiteReport Financial Report Extraction & Validation Agent

This application extracts, validates, and analyzes financial statements (Balance Sheet, Profit & Loss, Cash Flow) from PDF reports using LLMs and custom schemas. It supports uploading prompt/hint questions, semantic chunking, LLM-based validation, and feedback tracking.

## Features

- **PDF Extraction:** Extracts structured financial data from uploaded PDFs using Llama Cloud.
- **Custom Prompts:** Upload and manage validation questions and hints via CSV/Excel.
- **Semantic Search:** Finds the most relevant data chunks for each question using OpenAI embeddings.
- **LLM Validation:** Uses OpenAI GPT models to answer validation questions based on extracted data.
- **Feedback Loop:** Collects user feedback on LLM answers and tracks accuracy.
- **Visualization:** Displays feedback results and accuracy with tables and charts.

## Usage

1. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the app:**
    ```sh
    streamlit run llama_extract.py
    ```

3. **Workflow:**
    - Tab 1: Upload a CSV/Excel file with validation questions and hints.
    - Tab 2: Upload a financial report PDF, select a section, and validate using prompts.
    - Tab 3: Review LLM answer feedback and accuracy.

## File Structure

- `llama_extract.py`: Main Streamlit app.
- `prompts.db`: SQLite DB for storing prompts.
- `balance_sheet.pkl`, `pl_statement.pkl`, `cash_flow_statement.pkl`: Cached extracted data.
- `llm_feedback.json`: Stores LLM answer feedback.
- `prompts.csv`: Example prompts file.

## Environment Variables

API keys for Llama Cloud and OpenAI are set in the script. For production, use environment variables or a `.env` file.

## Requirements

See [requirements.txt](requirements.txt).
