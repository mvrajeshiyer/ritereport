import os
import uuid
import streamlit as st
import json
import re
import pickle
import numpy as np
from llama_cloud_services import LlamaExtract
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List
from llama_cloud.types import ExtractConfig, ExtractMode
import sqlite3
import pandas as pd
import plotly.express as px

from utils.secrets_manager import load_secrets

secrets = load_secrets()
os.environ["LLAMA_CLOUD_API_KEY"] = secrets["LLAMA_CLOUD_API_KEY"]
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

# --- Financial Statement Models (Pydantic) ---

class Signatory(BaseModel):
    name: str
    designation: str
    din: Optional[str] = None

class AuditorDetails(BaseModel):
    auditor_name: str
    firm_name: str
    registration_number: str
    place: Optional[str] = None
    date: Optional[str] = None

class CompanyMetadata(BaseModel):
    company_name: str
    statement_date: str
    place: Optional[str] = None
    directors: Optional[List[Signatory]] = None
    cfo: Optional[Signatory] = None
    company_secretary: Optional[Signatory] = None
    auditors: Optional[List[AuditorDetails]] = None

class CurrencyUnit(str, Enum):
    rupees = 'Rupees'
    lakhs = 'Lakhs'
    crores = 'Crores'
    millions = 'Millions'
    thousands = 'Thousands'

class LineItem(BaseModel):
    particular: str = Field(description="Name of the line item as per Division II")
    note_no: Optional[str] = Field(default=None, description="Reference note number in the financials")
    current_year: float = Field(description="Amount for current year")
    previous_year: float = Field(description="Amount for previous year")

class SectionTotal(BaseModel):
    current_year: float
    previous_year: float

class FinancialAssets(BaseModel):
    investments: Optional[List[LineItem]] = None
    trade_receivables: Optional[List[LineItem]] = None
    loans: Optional[List[LineItem]] = None
    others: Optional[List[LineItem]] = None

class NonCurrentAssets(BaseModel):
    property_plant_equipment: Optional[List[LineItem]] = None
    right_of_use_assets: Optional[List[LineItem]] = None 
    capital_work_in_progress: Optional[List[LineItem]] = None
    investment_property: Optional[List[LineItem]] = None
    goodwill: Optional[List[LineItem]] = None
    other_intangible_assets: Optional[List[LineItem]] = None
    intangible_assets_under_development: Optional[List[LineItem]] = None
    biological_assets_other_than_bearer_plants: Optional[List[LineItem]] = None
    financial_assets: Optional[FinancialAssets] = None
    deferred_tax_assets_net: Optional[List[LineItem]] = None
    other_non_current_assets: Optional[List[LineItem]] = None
    total: Optional[SectionTotal] = None

class CurrentFinancialAssets(BaseModel):
    investments: Optional[List[LineItem]] = None
    trade_receivables: Optional[List[LineItem]] = None
    cash_and_cash_equivalents: Optional[List[LineItem]] = None
    bank_balances_other_than_cash_and_cash_equivalents: Optional[List[LineItem]] = None
    loans: Optional[List[LineItem]] = None
    others: Optional[List[LineItem]] = None

class CurrentAssets(BaseModel):
    inventories: Optional[List[LineItem]] = None
    financial_assets: Optional[CurrentFinancialAssets] = None
    current_tax_assets_net: Optional[List[LineItem]] = None
    other_current_assets: Optional[List[LineItem]] = None
    total: Optional[SectionTotal] = None

class Assets(BaseModel):
    non_current_assets: NonCurrentAssets
    current_assets: CurrentAssets
    total_assets: SectionTotal

class TradePayables(BaseModel):
    micro_enterprises_and_small_enterprises: Optional[List[LineItem]] = None
    creditors_other_than_micro_enterprises_and_small_enterprises: Optional[List[LineItem]] = None

class NonCurrentFinancialLiabilities(BaseModel):
    borrowings: Optional[List[LineItem]] = None
    lease_liabilities: Optional[List[LineItem]] = None
    trade_payables: Optional[TradePayables] = None
    other_financial_liabilities: Optional[List[LineItem]] = None

class NonCurrentLiabilities(BaseModel):
    financial_liabilities: Optional[NonCurrentFinancialLiabilities] = None
    provisions: Optional[List[LineItem]] = None
    deferred_tax_liabilities_net: Optional[List[LineItem]] = None
    other_non_current_liabilities: Optional[List[LineItem]] = None
    total: Optional[SectionTotal] = None

class CurrentFinancialLiabilities(BaseModel):
    borrowings: Optional[List[LineItem]] = None
    lease_liabilities: Optional[List[LineItem]] = None
    trade_payables: Optional[TradePayables] = None
    other_financial_liabilities: Optional[List[LineItem]] = None

class CurrentLiabilities(BaseModel):
    financial_liabilities: Optional[CurrentFinancialLiabilities] = None
    other_current_liabilities: Optional[List[LineItem]] = None
    provisions: Optional[List[LineItem]] = None
    current_tax_liabilities_net: Optional[List[LineItem]] = None
    total: Optional[SectionTotal] = None

class EquityAndLiabilities(BaseModel):
    equity: Optional[List[LineItem]] = None
    non_current_liabilities: Optional[NonCurrentLiabilities] = None
    current_liabilities: Optional[CurrentLiabilities] = None
    total_equity_and_liabilities: Optional[SectionTotal] = None

class AccountingPolicyDetail(BaseModel):
    policy_name: str = Field(description="Name of the accounting policy")
    description: str = Field(description="Description of the policy")
    note_no: Optional[str] = Field(default=None, description="Reference note number if available")

class SignificantAccountingPolicies(BaseModel):
    policies: Optional[List[AccountingPolicyDetail]] = Field(
        default=None, description="List of significant accounting policies"
    )
    general_note: Optional[str] = Field(
        default=None, description="General note or introductory text for accounting policies"
    )

class AccompanyingNoteDetail(BaseModel):
    note_no: str = Field(description="Note number")
    title: Optional[str] = Field(default=None, description="Title or subject of the note")
    content: str = Field(description="Full text/content of the note")

class AccompanyingNotesStatement(BaseModel):
    notes: Optional[List[AccompanyingNoteDetail]] = Field(
        default=None, description="List of all accompanying notes"
    )
    general_note: Optional[str] = Field(
        default=None, description="General note or introductory text for the notes section"
    )

class BalanceSheet(BaseModel):
    metadata: CompanyMetadata
    unit: CurrencyUnit
    assets: Assets
    equity_and_liabilities: EquityAndLiabilities
    significant_accounting_policies: Optional[SignificantAccountingPolicies] = None
    accompanying_notes_statement: Optional[AccompanyingNotesStatement] = None
    board_of_directors_statement: Optional[str] = None

class PLLineItem(BaseModel):
    particular: str = Field(description="Name of the line item")
    note_no: Optional[str] = Field(default=None, description="Reference note number")
    current_period: float = Field(description="Amount for current reporting period")
    previous_period: float = Field(description="Amount for previous reporting period")

class ProfitAndLoss(BaseModel):
    metadata: CompanyMetadata
    unit: CurrencyUnit = Field(description="Unit of currency used in the report")
    revenue_from_operations: Optional[List[PLLineItem]] = None
    total_revenue_from_operations: Optional[PLLineItem] = None
    other_income: Optional[List[PLLineItem]] = None
    total_income: Optional[PLLineItem] = None
    expenses: Optional[List[PLLineItem]] = None
    total_expenses: Optional[PLLineItem] = None
    profit_before_exceptional_items_and_tax: Optional[PLLineItem] = None
    exceptional_items: Optional[PLLineItem] = None
    profit_before_tax: Optional[PLLineItem] = None
    tax_expense_current: Optional[PLLineItem] = None
    tax_expense_deferred: Optional[PLLineItem] = None
    profit_from_continuing_operations: Optional[PLLineItem] = None
    profit_from_discontinued_operations: Optional[PLLineItem] = None
    tax_expense_discontinued: Optional[PLLineItem] = None
    profit_after_tax_discontinued: Optional[PLLineItem] = None
    profit_for_the_period: Optional[PLLineItem] = None
    other_comprehensive_income_a: Optional[List[PLLineItem]] = None
    other_comprehensive_income_b: Optional[List[PLLineItem]] = None
    total_other_comprehensive_income: Optional[PLLineItem] = None
    total_comprehensive_income: Optional[PLLineItem] = None
    earnings_per_equity_share_continuing_basic: Optional[PLLineItem] = None
    earnings_per_equity_share_continuing_diluted: Optional[PLLineItem] = None
    earnings_per_equity_share_discontinued_basic: Optional[PLLineItem] = None
    earnings_per_equity_share_discontinued_diluted: Optional[PLLineItem] = None
    earnings_per_equity_share_total_basic: Optional[PLLineItem] = None
    earnings_per_equity_share_total_diluted: Optional[PLLineItem] = None

class CashFlowLineItem(BaseModel):
    particular: str = Field(description="Name of the cash flow line item")
    note_no: Optional[str] = Field(default=None, description="Reference note number")
    current_year: float = Field(description="Amount for current year")
    previous_year: float = Field(description="Amount for previous year")

class CashFlowStatement(BaseModel):
    metadata: CompanyMetadata
    unit: CurrencyUnit = Field(description="Unit of currency used in the report")
    operating_activities: List[CashFlowLineItem]
    investing_activities: List[CashFlowLineItem]
    financing_activities: List[CashFlowLineItem]
    net_increase_decrease_in_cash: Optional[CashFlowLineItem] = None
    cash_and_cash_equivalents_at_beginning: Optional[CashFlowLineItem] = None
    cash_and_cash_equivalents_at_end: Optional[CashFlowLineItem] = None

# --- Streamlit UI ---

st.set_page_config(page_title="Financial Report Agent", layout="wide")

st.title("📊 Financial Report Extraction & Validation Agent")

tab1, tab2, tab3 = st.tabs(["🗂️ Upload Prompts CSV", "🧾 Validation", "💬 Chat & Results"])

# --- TAB 1: Upload Prompts CSV ---
with tab1:
    st.header("Upload Prompt/Hint Questions (CSV/Excel)")
    uploaded_csv = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_csv:
        if uploaded_csv.name.endswith(".csv"):
            df = pd.read_csv(uploaded_csv)
        else:
            df = pd.read_excel(uploaded_csv)
        st.dataframe(df)
        # Always clear and reload prompts table
        conn = sqlite3.connect("prompts.db")
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS prompts;")
        conn.commit()
        df.to_sql("prompts", conn, if_exists="replace", index=False)
        st.success("Prompts and hints saved to SQLite database.")
        conn.close()
    else:
        st.info("Upload a CSV or Excel file with columns: section, question, hint")

# --- TAB 2: Validation (existing workflow, slightly refactored) ---
with tab2:
    st.header("Financial Statement Validation")
    uploaded_file = st.file_uploader("Upload Financial Report PDF", type=["pdf"])

    db_path = "prompts.db"
    def table_exists(db_path, table_name):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    if not os.path.exists(db_path) or not table_exists(db_path, "prompts"):
        st.warning("Please upload a valid prompts CSV in Tab 1 before using validation.")
    elif uploaded_file:
        # Section dropdown (only once!)
        section_options = ["Balance Sheet", "Profit & Loss", "Cash Flow"]
        selected_section = st.selectbox(
            "Select section to extract and embed:",
            section_options,
            key="validation_section_selectbox"
        )

        # Load questions from SQLite based on selected section
        conn = sqlite3.connect("prompts.db")
        query = f"SELECT question, hint FROM prompts WHERE section = ?"
        df_prompts = pd.read_sql_query(query, conn, params=(selected_section,))
        conn.close()

        st.subheader("Validation Questions")
        question_options = df_prompts["question"].tolist() + ["Other (type your own)"]
        selected_question_text = st.selectbox(
            "Select a question to validate:",
            question_options,
            key="validation_question_selectbox"
        )
        custom_question = ""

        def generate_hint_for_question(question):
            import openai
            prompt = (
            "You are a financial auditor. Given the following validation question for a financial statement, "
            "generate a short, actionable hint (1 sentence) to help an auditor answer it. "
            "Use accounting-specific terminology if possible.\n"
            f"Question: {question}\n"
            "Hint:"
            )
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()

        # --- Utility Functions for Saving/Loading Data ---

        def save_to_pkl(data, filename):
            """Save Python object to pickle file."""
            with open(filename, "wb") as f:
                pickle.dump(data, f)

        def load_from_pkl(filename):
            """Load Python object from pickle file."""
            with open(filename, "rb") as f:
                return pickle.load(f)

        def llm_answer(question, context, model="gpt-4o"):
            import openai
            prompt = (
            f"You are a financial auditor. Validate the financial report below and deliver a concise, bullet-point summary of your findings.\n\n"
            f"Report Data:\n{context}\n\n"
            f"Validation Question:\n{question}\n\n"
            f"Instructions:\n"
            f"- Use only the figures and text in “Report Data.” Do not infer or assume anything else.\n"
            f"- Answer in bullet points maximum of 4.\n"
            f"- If a calculation is needed, show it once with the formula in parentheses.\n"
            f"- Label every amount with its unit (e.g., ₹3.2 Crores, ₹45 Lakhs).\n"
            f"- Express percentages as “xx%.”\n"
            f"- At the end, give a single Final Answer: Valid” or Final Answer: Invalid” (with a one-sentence reason).\n"
            f"- Keep it crisp—CFO/Auditor should absorb it in under 30 seconds.\n"
            f"- Do not add anything extra or hallucinate figures.\n"
            f"- Provide only the answer which is relevant to the question. Dont provide any other additional answer\n"
        )



            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()

        def flatten_json_to_chunks(data, path=""):
            """
            Recursively flatten a nested JSON into text chunks for retrieval.
            Each chunk is a dict: {"text": ..., "path": ...}
            """
            chunks = []
            if isinstance(data, dict):
                for k, v in data.items():
                    new_path = f"{path}.{k}" if path else k
                    chunks.extend(flatten_json_to_chunks(v, new_path))
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    new_path = f"{path}[{idx}]"
                    chunks.extend(flatten_json_to_chunks(item, new_path))
            else:
                # Leaf node
                chunks.append({"text": f"{path}: {data}", "path": path})
            return chunks

        def section_level_chunks(data, parent_path=""):
            """
            Create section-level chunks for all top-level and second-level sections in the extracted JSON.
            Each chunk contains the JSON string of that section and its path.
            Adds human-readable labels for financial context.
            """
            # Comprehensive mapping for Indian financial statements
            field_labels = {
                # Metadata
                "metadata": "Company Metadata",
                "company_name": "Company Name",
                "statement_date": "Statement Date",
                "place": "Place",
                "directors": "Directors",
                "cfo": "Chief Financial Officer",
                "company_secretary": "Company Secretary",
                "auditors": "Auditors",
                # Balance Sheet
                "balance_sheet": "Balance Sheet",
                "assets": "Assets",
                "non_current_assets": "Non-Current Assets",
                "current_assets": "Current Assets",
                "property_plant_equipment": "Property, Plant and Equipment",
                "right_of_use_assets": "Right of Use Assets",
                "capital_work_in_progress": "Capital Work-in-Progress",
                "investment_property": "Investment Property",
                "goodwill": "Goodwill",
                "other_intangible_assets": "Other Intangible Assets",
                "intangible_assets_under_development": "Intangible Assets Under Development",
                "biological_assets_other_than_bearer_plants": "Biological Assets (Other than Bearer Plants)",
                "financial_assets": "Financial Assets",
                "investments": "Investments",
                "trade_receivables": "Trade Receivables",
                "loans": "Loans",
                "others": "Other Financial Assets",
                "deferred_tax_assets_net": "Deferred Tax Assets (Net)",
                "other_non_current_assets": "Other Non-Current Assets",
                "inventories": "Inventories",
                "current_tax_assets_net": "Current Tax Assets (Net)",
                "other_current_assets": "Other Current Assets",
                "total_assets": "Total Assets",
                # Liabilities & Equity
                "equity_and_liabilities": "Equity and Liabilities",
                "equity": "Equity",
                "share_capital": "Share Capital",
                "other_equity": "Other Equity",
                "non_current_liabilities": "Non-Current Liabilities",
                "current_liabilities": "Current Liabilities",
                "financial_liabilities": "Financial Liabilities",
                "borrowings": "Borrowings",
                "lease_liabilities": "Lease Liabilities",
                "trade_payables": "Trade Payables",
                "micro_enterprises_and_small_enterprises": "Micro Enterprises and Small Enterprises",
                "creditors_other_than_micro_enterprises_and_small_enterprises": "Other Creditors",
                "other_financial_liabilities": "Other Financial Liabilities",
                "provisions": "Provisions",
                "deferred_tax_liabilities_net": "Deferred Tax Liabilities (Net)",
                "other_non_current_liabilities": "Other Non-Current Liabilities",
                "other_current_liabilities": "Other Current Liabilities",
                "current_tax_liabilities_net": "Current Tax Liabilities (Net)",
                "total_equity_and_liabilities": "Total Equity and Liabilities",
                # P&L
                "profit_and_loss": "Profit and Loss Statement",
                "revenue_from_operations": "Revenue from Operations",
                "total_revenue_from_operations": "Total Revenue from Operations",
                "other_income": "Other Income",
                "total_income": "Total Income",
                "expenses": "Expenses",
                "total_expenses": "Total Expenses",
                "profit_before_exceptional_items_and_tax": "Profit Before Exceptional Items and Tax",
                "exceptional_items": "Exceptional Items",
                "profit_before_tax": "Profit Before Tax",
                "tax_expense_current": "Current Tax Expense",
                "tax_expense_deferred": "Deferred Tax Expense",
                "profit_from_continuing_operations": "Profit from Continuing Operations",
                "profit_from_discontinued_operations": "Profit from Discontinued Operations",
                "tax_expense_discontinued": "Tax Expense on Discontinued Operations",
                "profit_after_tax_discontinued": "Profit After Tax from Discontinued Operations",
                "profit_for_the_period": "Profit for the Period",
                "other_comprehensive_income_a": "Other Comprehensive Income (Section A)",
                "other_comprehensive_income_b": "Other Comprehensive Income (Section B)",
                "total_other_comprehensive_income": "Total Other Comprehensive Income",
                "total_comprehensive_income": "Total Comprehensive Income",
                "earnings_per_equity_share_continuing_basic": "Earnings per Equity Share (Continuing, Basic)",
                "earnings_per_equity_share_continuing_diluted": "Earnings per Equity Share (Continuing, Diluted)",
                "earnings_per_equity_share_discontinued_basic": "Earnings per Equity Share (Discontinued, Basic)",
                "earnings_per_equity_share_discontinued_diluted": "Earnings per Equity Share (Discontinued, Diluted)",
                "earnings_per_equity_share_total_basic": "Earnings per Equity Share (Total, Basic)",
                "earnings_per_equity_share_total_diluted": "Earnings per Equity Share (Total, Diluted)",
                # Cash Flow
                "cash_flow_statement": "Cash Flow Statement",
                "operating_activities": "Cash Flow from Operating Activities",
                "investing_activities": "Cash Flow from Investing Activities",
                "financing_activities": "Cash Flow from Financing Activities",
                "net_increase_decrease_in_cash": "Net Increase/Decrease in Cash",
                "cash_and_cash_equivalents_at_beginning": "Cash and Cash Equivalents at Beginning",
                "cash_and_cash_equivalents_at_end": "Cash and Cash Equivalents at End",
                # Notes & Policies
                "significant_accounting_policies": "Significant Accounting Policies",
                "accompanying_notes_statement": "Accompanying Notes Statement",
                "board_of_directors_statement": "Board of Directors Statement",
                # Add more as needed for your data model
            }
            chunks = []
            if isinstance(data, dict):
                for k, v in data.items():
                    path = f"{parent_path}.{k}" if parent_path else k
                    label = field_labels.get(k, k.replace("_", " ").title())
                    if isinstance(v, (dict, list)):
                        chunks.append({
                            "text": f"{label} ({path}): {json.dumps(v, indent=2)}",
                            "path": path
                        })
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                sub_path = f"{path}.{kk}"
                                sub_label = field_labels.get(kk, kk.replace("_", " ").title())
                                if isinstance(vv, (dict, list)):
                                    chunks.append({
                                        "text": f"{sub_label} ({sub_path}): {json.dumps(vv, indent=2)}",
                                        "path": sub_path
                                    })
            return chunks

        def get_top_k_chunks(question, chunks, k=5, metadata_boost=0.15, generic_boost=0.15):
            import re
            embed_model = OpenAIEmbedding()
            question_emb = np.array(embed_model.get_text_embedding(question)).astype('float32')
            chunk_embs = [embed_model.get_text_embedding(chunk["text"]) for chunk in chunks]
            chunk_embs = np.array(chunk_embs).astype('float32')
            sims = np.dot(chunk_embs, question_emb) / (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(question_emb) + 1e-8)

            # Lowercase and normalize question for matching
            q_lower = question.lower()
            # Remove punctuation and split into words
            q_words = re.findall(r'\w+', q_lower)
            # Generate all possible contiguous phrases up to 4 words (adjust as needed)
            phrases = set()
            for n in range(1, min(5, len(q_words)+1)):
                for i in range(len(q_words)-n+1):
                    phrase = " ".join(q_words[i:i+n])
                    # Normalize plural to singular (very basic)
                    if phrase.endswith('ies'):
                        phrase = phrase[:-3] + 'y'
                    elif phrase.endswith('s') and not phrase.endswith('ss'):
                        phrase = phrase[:-1]
                    phrases.add(phrase)

            for i, chunk in enumerate(chunks):
                path_lower = chunk["path"].lower()
                text_lower = chunk["text"].lower()
                # Always boost metadata
                if "metadata" in path_lower:
                    sims[i] += metadata_boost
                # Phrase-aware boosting: boost if any phrase from question is in chunk text or path
                for phrase in phrases:
                    if phrase and (phrase in text_lower or phrase in path_lower):
                        sims[i] += generic_boost
                        break  # Only boost once per chunk

            # Get top-k unique chunks
            top_indices = np.argsort(sims)[::-1]
            seen = set()
            unique_chunks = []
            for idx in top_indices:
                text = chunks[idx]["text"]
                if text not in seen:
                    seen.add(text)
                    unique_chunks.append(chunks[idx])
                if len(unique_chunks) == k:
                    break
            return unique_chunks

        def rerank_chunks(question, chunks, model="gpt-3.5-turbo"):
            """
            Rerank chunks using OpenAI LLM by asking which are most relevant to the question.
            Returns the chunks sorted by relevance (most relevant first).
            """
            import openai
            # Prepare the prompt for reranking
            chunk_texts = [f"{i+1}. {chunk['text']}" for i, chunk in enumerate(chunks)]
            prompt = (
                f"Given the following question:\n{question}\n\n"
                f"And the following extracted information chunks:\n"
                f"{chr(10).join(chunk_texts)}\n\n"
                "Rank the chunks from most relevant to least relevant for answering the question. "
                "Return a list of chunk numbers in order of relevance, most relevant first."
            )
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            # Parse the response to get the order
            import re
            order = re.findall(r'\d+', response.choices[0].message.content)
            order = [int(i)-1 for i in order if int(i)-1 < len(chunks)]
            # Return chunks in reranked order
            return [chunks[i] for i in order]

        import re

        def display_llm_answer(answer):
            # Find lines that look like formulas (contain '=' and at least one operator)
            formula_lines = []
            other_lines = []
            for line in answer.split('\n'):
                if '=' in line and any(op in line for op in ['+', '-', '*', '/', '%']):
                    formula_lines.append(line.strip())
                else:
                    other_lines.append(line)
            # Display formulas as code, rest as markdown
            if other_lines:
                st.markdown('\n'.join(other_lines))
            for formula in formula_lines:
                st.code(formula, language="python")

        # --- Main Workflow ---
        if uploaded_file:
            # Save uploaded PDF to disk
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.read())

            llama_extract = LlamaExtract()

            # Section-specific prompt and schema
            if selected_section == "Balance Sheet":
                prompt = """
                Extract only the balance sheet section from the PDF. 
                In addition to the balance sheet data, also extract:
                - Company name
                - Statement date
                - Place
                - Directors details (name, designation, DIN)
                - CFO details (name, designation, DIN)
                - Company secretary details (name, designation, DIN)
                - Auditors details (auditor name, firm name, registration number, place, date)
                - Significant accounting policies
                - The accompanying notes are an integral part of the Ind AS financial statements
                - The full phrase containing 'for and on behalf of Board of directors of'
                Structure the data as per the BalanceSheet schema, including all metadata fields and the above additional extracted text as separate fields.
                """
                schema = BalanceSheet
                pkl_file = "balance_sheet.pkl"
                tab_label = "Balance Sheet"
                key_prefix = "bs"
            elif selected_section == "Profit & Loss":
                prompt = """
                Extract only the profit and loss statement section from the PDF. 
                In addition to the P&L data, also extract:
                - Company name
                - Statement date
                - Place
                - Directors details (name, designation, DIN)
                - CFO details (name, designation, DIN)
                - Company secretary details (name, designation, DIN)
                - Auditors details (auditor name, firm name, registration number, place, date)
                Structure the data as per the ProfitAndLoss schema, including all metadata fields.
                """
                schema = ProfitAndLoss
                pkl_file = "pl_statement.pkl"
                tab_label = "Profit and Loss"
                key_prefix = "pl"
            else:
                prompt = """
                Extract only the cash flow statement section from the PDF. 
                In addition to the cash flow data, also extract:
                - Company name
                - Statement date
                - Place
                - Directors details (name, designation, DIN)
                - CFO details (name, designation, DIN)
                - Company secretary details (name, designation, DIN)
                - Auditors details (auditor name, firm name, registration number, place, date)
                Structure the data as per the CashFlowStatement schema, including all metadata fields.
                """
                schema = CashFlowStatement
                pkl_file = "cash_flow_statement.pkl"
                tab_label = "Cash Flow"
                key_prefix = "cf"

            # --- Extraction & Caching ---
            if os.path.exists(pkl_file):
                data = load_from_pkl(pkl_file)
                st.info(f"Loaded {tab_label} data from cache.")
            else:
                config = ExtractConfig(
                    use_reasoning=False,
                    cite_sources=False,
                    extraction_mode=ExtractMode.FAST,
                    prompt=prompt
                )
                agent = llama_extract.create_agent(
                    name=f"{key_prefix}_agent_{str(uuid.uuid4())[:8]}",
                    data_schema=schema,
                    config=config
                )
                result = agent.extract("temp_upload.pdf")
                data = getattr(result, "data", {})
                save_to_pkl(data, pkl_file)
                # Delete the agent after extraction to free resources
                try:
                    llama_extract.delete_agent(name=agent.name)
                except Exception:
                    pass
                st.success(f"{tab_label} extracted, cached, and agent deleted.")


            def generate_hint_for_question(question):
                import openai
                prompt = (
                "You are a financial auditor. Given the following validation question for a financial statement, "
                "generate a short, actionable hint (1 sentence) to help an auditor answer it. "
                "Use accounting-specific terminology if possible.\n"
                f"Question: {question}\n"
                "Hint:"
                )
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                return response.choices[0].message.content.strip()


            if selected_question_text == "Other (type your own)":
                custom_question = st.text_input("Type your custom validation question:")
                if custom_question:
                    auto_hint = generate_hint_for_question(custom_question)
                else:
                    auto_hint = ""
                final_question = custom_question
                selected_question_obj = {"question": custom_question, "hint": auto_hint}
            else:
                selected_row = df_prompts[df_prompts["question"] == selected_question_text].iloc[0]
                selected_question_obj = {"question": selected_row["question"], "hint": selected_row["hint"]}
                final_question = selected_question_obj["question"]

            if st.button("Validate Selected Question"):
                with st.spinner("Retrieving top relevant chunks..."):
                    # 1. Section-level chunking
                    chunks = section_level_chunks(data)
                    
                    with st.expander("Extracted Balance Sheet (JSON)"):
                        st.json(data)

                    with st.expander("Top 5 Relevant Chunks (Semantic Search)"):
                        # Get top 5 relevant chunks
                        top_chunks = get_top_k_chunks(selected_question_text, chunks, k=10)

                        # Rerank the top 5 chunks
                        reranked_chunks = rerank_chunks(selected_question_text, top_chunks)

                        # Use the top 3 reranked chunks for LLM context
                        top3_chunks = reranked_chunks[:5]
                        context = "\n".join([chunk["text"] for chunk in top3_chunks])

                        for i, chunk in enumerate(top_chunks):
                            st.code(f"{i+1}. {chunk['text']} (path: {chunk['path']})")

                    with st.expander("Reranked Top 5 Chunks"):
                        for i, chunk in enumerate(reranked_chunks):
                            st.code(f"{i+1}. {chunk['text']} (path: {chunk['path']})")

                    # 2. Prepare context for LLM (top 3 only)
                    top3_chunks = top_chunks[:10]
                    context = "\n".join([chunk["text"] for chunk in top3_chunks])
                    with st.expander("LLM Context (to be sent to LLM)"):
                        st.code(context)
                    st.session_state.llm_context = context
                    st.session_state.llm_question = selected_question_text
                llm_query = f"{selected_question_obj['question']}\nHint: {selected_question_obj['hint']}"
                st.session_state.llm_resp = llm_answer(llm_query, context)

        # --- Compute and display LLM answer validation accuracy ---
        def compute_accuracy(feedback_file="llm_feedback.json"):
            if not os.path.exists(feedback_file):
                return 0, 0, 0.0
            with open(feedback_file, "r") as f:
                feedbacks = json.load(f)
            total = len(feedbacks)
            correct = sum(1 for fb in feedbacks if fb["feedback"] == "Correct")
            accuracy = correct / total if total else 0
            return correct, total, accuracy

        import re

        def display_llm_answer(answer):
            # Find lines that look like formulas (contain '=' and at least one operator)
            formula_lines = []
            other_lines = []
            for line in answer.split('\n'):
                if '=' in line and any(op in line for op in ['+', '-', '*', '/', '%']):
                    formula_lines.append(line.strip())
                else:
                    other_lines.append(line)
            # Display formulas as code, rest as markdown
            if other_lines:
                st.markdown('\n'.join(other_lines))
            for formula in formula_lines:
                st.code(formula, language="python")

        if 'llm_resp' not in st.session_state:
            st.session_state.llm_resp = None
        if 'llm_context' not in st.session_state:
            st.session_state.llm_context = None
        if 'llm_question' not in st.session_state:
            st.session_state.llm_question = None

        if st.session_state.llm_resp:
            with st.expander("LLM Answer"):
                display_llm_answer(st.session_state.llm_resp)

            # --- Feedback UI ---
            llm_feedback_key = f"llm_feedback_{selected_section.replace(' ', '_')}"
            llm_feedback_state_key = f"llm_feedback_submitted_{selected_section.replace(' ', '_')}"
            if llm_feedback_key not in st.session_state:
                st.session_state[llm_feedback_key] = "Not validated"
            if llm_feedback_state_key not in st.session_state:
                st.session_state[llm_feedback_state_key] = False

            llm_feedback = st.radio(
                "Is the LLM answer correct?",
                ("Not validated", "Correct", "Incorrect"),
                key=llm_feedback_key,
                #disabled=st.session_state[llm_feedback_state_key]
            )

            if llm_feedback != "Not validated" and not st.session_state[llm_feedback_state_key]:
                feedback_data = {
                    "question": final_question,  # This is the actual question, typed or selected
                    "llm_answer": st.session_state.llm_resp,
                    "chunks_used": [st.session_state.llm_context],
                    "feedback": llm_feedback
                }
                feedback_file = "llm_feedback.json"
                if os.path.exists(feedback_file):
                    with open(feedback_file, "r") as f:
                        all_feedback = json.load(f)
                else:
                    all_feedback = []
                all_feedback.append(feedback_data)
                with open(feedback_file, "w") as f:
                    json.dump(all_feedback, f, indent=2)
                st.session_state[llm_feedback_state_key] = True
                st.success("LLM Answer Feedback submitted!")

            # --- Show accuracy only after LLM response ---
            correct, total, accuracy = compute_accuracy()
            st.info(f"LLM Answer Validation Accuracy: {correct}/{total} ({accuracy:.0%})")

# --- TAB 3: Chat & Results ---
import plotly.express as px

with tab3:
    st.header("LLM Feedback & Validation Results")

    # Load feedback data
    feedback_file = "llm_feedback.json"
    if not os.path.exists(feedback_file):
        st.info("No feedback data found yet.")
    else:
        with open(feedback_file, "r") as f:
            feedbacks = json.load(f)

        # Prepare DataFrame for display
        import pandas as pd
        df_feedback = pd.DataFrame(feedbacks)
        # Only keep relevant columns and clean up
        df_feedback_display = df_feedback[["question", "llm_answer", "feedback"]].copy()
        df_feedback_display["feedback"] = df_feedback_display["feedback"].replace({"Correct": "✅ Correct", "Incorrect": "❌ Incorrect"})

        st.subheader("Validation Feedback Table")
        st.dataframe(df_feedback_display, use_container_width=True)

        # Pie chart for feedback summary
        correct_count = (df_feedback["feedback"] == "Correct").sum()
        incorrect_count = (df_feedback["feedback"] == "Incorrect").sum()
        total = correct_count + incorrect_count

        if total > 0:
            pie_df = pd.DataFrame({
                "Result": ["Correct", "Incorrect"],
                "Count": [correct_count, incorrect_count]
            })
            fig = px.pie(pie_df, names="Result", values="Count", color="Result",
                         color_discrete_map={"Correct": "green", "Incorrect": "red"},
                         title="LLM Validation Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No correct/incorrect feedback available yet.")

        # Optionally, show details for a selected question
        st.subheader("Detailed LLM Answer")

        # Build a display list that always shows the actual question (even for custom-typed)
        question_display_list = []
        for idx, row in df_feedback.iterrows():
            q = row["question"]
            # If the question is empty or "Other (type your own)", show the actual custom question from the feedback
            if not q or q.strip().lower() == "other (type your own)":
                q = row.get("question", "Custom Question")
            question_display_list.append(q)

        selected_question_display = st.selectbox(
            "Select a question to view the LLM answer:",
            question_display_list,
            key="feedback_question_selectbox"
        )

        # Find the feedback row matching the selected question (by index, which is robust for custom questions)
        selected_idx = question_display_list.index(selected_question_display)
        selected_feedback = df_feedback.iloc[selected_idx]

        answer = selected_feedback["llm_answer"]
        st.markdown(f"**LLM Answer:**\n\n{answer}")

# Optional: Hide Streamlit default menu/footer for cleaner look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
