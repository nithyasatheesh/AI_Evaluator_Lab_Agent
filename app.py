import io
import json
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from docx import Document
from openai import OpenAI, OpenAIError
from PyPDF2 import PdfReader

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Project Evaluator",
    page_icon="📊",
    layout="wide",
)

st.title("📊 PDF-Based Knowledge Base RAG — Submission Evaluator")
st.caption(
    "Evaluates student submissions against the uploaded problem statement and "
    "5-task, 100-mark rubric."
)

MODEL = "gpt-4.1"
MAX_WORKERS = 4

MAX_CONTEXT = {
    "documentation": 14000,
    "notebooks": 18000,
    "code": 24000,
    "other": 10000,
}

SUPPORTED_SINGLE = {
    ".ipynb", ".py", ".html", ".htm", ".md", ".txt",
    ".docx", ".pdf", ".json", ".yaml", ".yml", ".sql", ".csv"
}

# ============================================================
# OPENAI
# ============================================================

def get_client() -> Optional[OpenAI]:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OPENAI_API_KEY is missing. Add it under Streamlit "
            "Secrets before running the evaluator."
        )
        return None
    return OpenAI(api_key=api_key, timeout=180, max_retries=2)


client = get_client()

# ============================================================
# TEXT READERS
# ============================================================

def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode(errors="ignore")


def read_pdf(raw: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[PDF PAGE {i}]\n{text}")
        return "\n\n".join(pages)
    except Exception as exc:
        logging.exception("PDF parsing failed")
        return f"[PDF PARSE ERROR] {exc}"


def read_docx(raw: bytes) -> str:
    try:
        doc = Document(io.BytesIO(raw))
        parts = []

        for p in doc.paragraphs:
            if p.text.strip():
                parts.append(p.text)

        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))

        return "\n".join(parts)
    except Exception as exc:
        logging.exception("DOCX parsing failed")
        return f"[DOCX PARSE ERROR] {exc}"


def read_html(raw: bytes) -> str:
    soup = BeautifulSoup(decode_bytes(raw), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def read_notebook(raw: bytes) -> str:
    try:
        nb = nbformat.reads(decode_bytes(raw), as_version=4)
    except Exception as exc:
        return f"[NOTEBOOK PARSE ERROR] {exc}"

    parts = []
    for i, cell in enumerate(nb.cells, start=1):
        source = "".join(cell.get("source", "")).strip()
        if not source:
            continue

        if cell.cell_type == "markdown":
            parts.append(f"[MARKDOWN CELL {i}]\n{source}")

        elif cell.cell_type == "code":
            parts.append(f"[CODE CELL {i}]\n{source}")

            outputs = cell.get("outputs", [])
            if outputs:
                output_text = []
                for output in outputs:
                    if "text" in output:
                        output_text.append("".join(output["text"]))
                    elif "data" in output:
                        data = output["data"]
                        if "text/plain" in data:
                            output_text.append("".join(data["text/plain"]))
                if output_text:
                    parts.append(
                        f"[OUTPUT CELL {i}]\n" +
                        "\n".join(output_text)
                    )

    return "\n\n".join(parts)


def read_text(raw: bytes) -> str:
    return decode_bytes(raw)


# ============================================================
# SUBMISSION PARSER
# ============================================================

def empty_result() -> Dict[str, Any]:
    return {
        "documentation": [],
        "notebooks": [],
        "code": [],
        "other": [],
        "files": [],
        "warnings": [],
    }


def add_file(result: Dict[str, Any], filename: str, raw: bytes) -> None:
    suffix = Path(filename).suffix.lower()

    result["files"].append(filename)

    try:
        if suffix == ".ipynb":
            text = read_notebook(raw)
            result["notebooks"].append(
                f"FILE: {filename}\n{text}"
            )

        elif suffix == ".py":
            result["code"].append(
                f"FILE: {filename}\n{decode_bytes(raw)}"
            )

        elif suffix in {".pdf"}:
            result["documentation"].append(
                f"FILE: {filename}\n{read_pdf(raw)}"
            )

        elif suffix == ".docx":
            result["documentation"].append(
                f"FILE: {filename}\n{read_docx(raw)}"
            )

        elif suffix in {".html", ".htm"}:
            result["documentation"].append(
                f"FILE: {filename}\n{read_html(raw)}"
            )

        elif suffix in {".md", ".txt"}:
            result["documentation"].append(
                f"FILE: {filename}\n{read_text(raw)}"
            )

        elif suffix in {".json", ".yaml", ".yml", ".sql", ".csv"}:
            result["other"].append(
                f"FILE: {filename}\n{decode_bytes(raw)}"
            )

        else:
            result["warnings"].append(
                f"Unsupported file inside submission: {filename}"
            )

    except Exception as exc:
        logging.exception("Could not parse %s", filename)
        result["warnings"].append(
            f"Could not parse {filename}: {exc}"
        )


def parse_zip(raw: bytes) -> Dict[str, Any]:
    result = empty_result()

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue

                # Avoid hidden/cache artifacts
                name = info.filename
                if (
                    "__pycache__" in name
                    or name.startswith(".")
                    or "/." in name
                ):
                    continue

                add_file(result, name, z.read(info))

    except zipfile.BadZipFile:
        result["warnings"].append("Uploaded ZIP is not a valid ZIP archive.")

    return result


def parse_submission(uploaded_file) -> Dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix == ".zip":
        return parse_zip(raw)

    result = empty_result()
    add_file(result, uploaded_file.name, raw)
    return result


def truncate_sections(items: List[str], limit: int) -> str:
    text = "\n\n".join(x for x in items if x.strip())
    return text[:limit]


def build_submission_context(parsed: Dict[str, Any]) -> str:
    return f"""
SUBMISSION FILES
{chr(10).join(parsed["files"])}

DOCUMENTATION
{truncate_sections(parsed["documentation"], MAX_CONTEXT["documentation"])}

NOTEBOOKS
{truncate_sections(parsed["notebooks"], MAX_CONTEXT["notebooks"])}

CODE
{truncate_sections(parsed["code"], MAX_CONTEXT["code"])}

OTHER FILES
{truncate_sections(parsed["other"], MAX_CONTEXT["other"])}

PARSER WARNINGS
{chr(10).join(parsed["warnings"])}
"""


# ============================================================
# PROBLEM + RUBRIC
# ============================================================

def read_problem(uploaded_problem) -> str:
    if uploaded_problem is None:
        return ""

    suffix = Path(uploaded_problem.name).suffix.lower()
    raw = uploaded_problem.getvalue()

    if suffix == ".pdf":
        return read_pdf(raw)
    if suffix == ".docx":
        return read_docx(raw)
    if suffix in {".txt", ".md"}:
        return decode_bytes(raw)

    return ""


RUBRIC_ALIASES = {
    "criterion": [
        "criterion",
        "criteria",
        "evaluation criteria",
    ],
    "max_score": [
        "max score",
        "max marks",
        "marks",
        "score",
        "weight",
    ],
    "description": [
        "description",
        "evaluation parameters",
        "details",
        "evaluation description",
    ],
}


def norm_col(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def find_column(columns, aliases):
    lookup = {norm_col(c): c for c in columns}
    for alias in aliases:
        if norm_col(alias) in lookup:
            return lookup[norm_col(alias)]
    return None


def normalize_rubric(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    mapping = {}
    missing = []

    for canonical, aliases in RUBRIC_ALIASES.items():
        col = find_column(df.columns, aliases)
        if col is None:
            missing.append(canonical)
        else:
            mapping[col] = canonical

    if missing:
        return None, missing

    normalized = df.rename(columns=mapping)

    # Keep only the three fields needed by evaluator.
    normalized = normalized[
        ["criterion", "max_score", "description"]
    ]

    normalized = normalized.dropna(
        subset=["criterion", "max_score"],
        how="any"
    )

    normalized["criterion"] = (
        normalized["criterion"]
        .astype(str)
        .str.strip()
    )

    normalized["description"] = (
        normalized["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    normalized["max_score"] = pd.to_numeric(
        normalized["max_score"],
        errors="coerce"
    )

    normalized = normalized.dropna(subset=["max_score"])

    return normalized, []


def rubric_to_text(df: pd.DataFrame) -> str:
    lines = []

    for _, row in df.iterrows():
        lines.append(
            f"""
Criterion: {row['criterion']}
Maximum Score: {row['max_score']}
Evaluation Description: {row['description']}
"""
        )

    return "\n".join(lines)


# ============================================================
# EVALUATION PROMPT
# ============================================================

def build_prompt(
    problem_text: str,
    rubric_text: str,
    submission_context: str,
    custom_instructions: str,
) -> str:

    return f"""
You are evaluating a student's implementation for a guided project.

IMPORTANT:
- Evaluate ONLY against the supplied problem statement and rubric.
- Do NOT give credit for claims that are not supported by submission evidence.
- Code, notebook cells, documentation, outputs, and configuration files are evidence.
- If a feature is described but not implemented, do not award implementation marks.
- If code is present but clearly incorrect/nonfunctional, deduct accordingly.
- Do not infer successful execution merely because code exists.
- Do not assume an API call worked unless evidence supports it.
- Do not reward unrelated functionality.
- Do not exceed the maximum score for any criterion.
- No evidence means no score for that criterion.
- Be strict but fair.
- The final score must be based on the 5 rubric criteria.
- Check the student's implementation against the actual technical requirements.

PROBLEM STATEMENT
=================
{problem_text}

RUBRIC
======
{rubric_text}

ADDITIONAL STRICT INSTRUCTIONS
==============================
{custom_instructions or "Evaluate strictly from the supplied problem statement, rubric, and submission evidence."}

STUDENT SUBMISSION
==================
{submission_context}

SPECIAL TECHNICAL CHECKS FOR THIS PROJECT
=========================================
Verify evidence for the following where applicable:

1. PDF source/input
2. LangChain PDF loader
3. Fixed-size text chunking
4. Chunk size = 1000
5. Configurable chunk overlap
6. text-embedding-3-small
7. ChromaDB
8. Retrieval / Top-K relevant chunks
9. GPT-4o-mini
10. Context construction from retrieved chunks
11. Answer generation based only on retrieved context
12. Handling of questions not answered by the PDF
13. User PDF upload/provision in the final application
14. End-to-end pipeline
15. Source/page display
16. README/documentation
17. Code quality and implementation completeness

SCORING
=======
Return a numeric score for EVERY rubric criterion using the exact criterion names from the rubric.

For each score:
- Award only marks supported by evidence.
- Use the full available range where justified.
- Do not automatically give partial credit simply because a concept is mentioned.
- Distinguish between explanation and implementation.
- A hard-coded demonstration is not equivalent to a generic implementation.
- A notebook containing only copied solution code without evidence of understanding should not automatically receive full marks.
- Missing required components must reduce the relevant task score.

OUTPUT FORMAT
=============
Return ONLY valid JSON:

{{
  "scores": {{
    "exact rubric criterion name": numeric_score
  }},
  "criterion_feedback": {{
    "exact rubric criterion name": "Specific evidence-based explanation of the score."
  }},
  "qualitative_feedback": {{
    "overall_feedback": "",
    "language_feedback": "",
    "analysis_feedback": "",
    "clarity_feedback": ""
  }},
  "strengths": [],
  "improvements": [],
  "missing_requirements": [],
  "evidence_summary": []
}}

The criterion names in "scores" and "criterion_feedback" MUST exactly match the rubric.
"""


# ============================================================
# SAFE JSON / SCORING
# ============================================================

def parse_json(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return {
                "scores": {},
                "criterion_feedback": {},
                "qualitative_feedback": {
                    "overall_feedback": "Model response was not valid JSON."
                },
                "strengths": [],
                "improvements": ["Model response could not be parsed."],
                "missing_requirements": [],
                "evidence_summary": [],
            }

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {
                "scores": {},
                "criterion_feedback": {},
                "qualitative_feedback": {
                    "overall_feedback": "Model response contained invalid JSON."
                },
                "strengths": [],
                "improvements": ["Model response could not be parsed."],
                "missing_requirements": [],
                "evidence_summary": [],
            }

    if not isinstance(data, dict):
        data = {}

    data.setdefault("scores", {})
    data.setdefault("criterion_feedback", {})
    data.setdefault("qualitative_feedback", {})
    data.setdefault("strengths", [])
    data.setdefault("improvements", [])
    data.setdefault("missing_requirements", [])
    data.setdefault("evidence_summary", [])

    return data


def normalize_score(value: Any, maximum: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(score, maximum))


def score_lookup(scores: Dict[str, Any], criterion: str, maximum: float) -> float:
    if criterion in scores:
        return normalize_score(scores[criterion], maximum)

    normalized = {
        norm_col(k): v
        for k, v in scores.items()
    }

    return normalize_score(
        normalized.get(norm_col(criterion), 0),
        maximum
    )


def text_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(x) for x in value if str(x).strip())
    return str(value or "")


# ============================================================
# OPENAI EVALUATION
# ============================================================

def evaluate_with_llm(prompt: str) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("OpenAI client is not configured.")

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict academic/project evaluator. "
                    "Award marks only for evidence in the submission. "
                    "Use the supplied rubric as the sole scoring authority."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    return parse_json(content)


# ============================================================
# PROCESS ONE STUDENT
# ============================================================

def process_submission(
    uploaded_file,
    problem_text: str,
    rubric_df: pd.DataFrame,
    rubric_text: str,
    custom_instructions: str,
) -> Dict[str, Any]:

    parsed = parse_submission(uploaded_file)

    result = {
        "Participant": uploaded_file.name,
    }

    context = build_submission_context(parsed)

    if not any(
        parsed[k]
        for k in ["documentation", "notebooks", "code", "other"]
    ):
        evaluation = {
            "scores": {},
            "criterion_feedback": {},
            "qualitative_feedback": {
                "overall_feedback": "No readable submission evidence was found."
            },
            "strengths": [],
            "improvements": ["Submit a readable IPYNB, ZIP, DOCX, PDF, HTML, or supported code file."],
            "missing_requirements": [],
            "evidence_summary": [],
        }
    else:
        prompt = build_prompt(
            problem_text,
            rubric_text,
            context,
            custom_instructions,
        )

        try:
            evaluation = evaluate_with_llm(prompt)
        except (OpenAIError, TimeoutError, RuntimeError, ValueError) as exc:
            logging.exception("Evaluation failed for %s", uploaded_file.name)
            evaluation = {
                "scores": {},
                "criterion_feedback": {},
                "qualitative_feedback": {
                    "overall_feedback": f"Evaluation failed: {exc}"
                },
                "strengths": [],
                "improvements": [str(exc)],
                "missing_requirements": [],
                "evidence_summary": [],
            }

    scores = evaluation.get("scores", {})
    criterion_feedback = evaluation.get("criterion_feedback", {})

    total = 0.0
    maximum_total = 0.0

    for _, row in rubric_df.iterrows():
        criterion = str(row["criterion"]).strip()
        maximum = float(row["max_score"])

        score = score_lookup(scores, criterion, maximum)

        result[criterion] = score
        result[f"{criterion} - Feedback"] = str(
            criterion_feedback.get(
                criterion,
                criterion_feedback.get(norm_col(criterion), "")
            )
        )

        total += score
        maximum_total += maximum

    result["Total"] = round(total, 2)
    result["Max Total"] = round(maximum_total, 2)
    result["Percentage"] = (
        round((total / maximum_total) * 100, 2)
        if maximum_total else 0
    )

    if total >= 90:
        result["Rating"] = "Excellent"
    elif total >= 75:
        result["Rating"] = "Good"
    elif total >= 60:
        result["Rating"] = "Satisfactory"
    else:
        result["Rating"] = "Needs Improvement"

    qf = evaluation.get("qualitative_feedback", {})
    if isinstance(qf, dict):
        result["Overall Feedback"] = str(qf.get("overall_feedback", ""))
        result["Language Feedback"] = str(qf.get("language_feedback", ""))
        result["Analysis Feedback"] = str(qf.get("analysis_feedback", ""))
        result["Clarity Feedback"] = str(qf.get("clarity_feedback", ""))
    else:
        result["Overall Feedback"] = str(qf or "")

    result["Strengths"] = text_list(evaluation.get("strengths"))
    result["Improvements"] = text_list(evaluation.get("improvements"))
    result["Missing Requirements"] = text_list(
        evaluation.get("missing_requirements")
    )
    result["Evidence Summary"] = text_list(
        evaluation.get("evidence_summary")
    )
    result["Parser Warnings"] = text_list(parsed.get("warnings"))

    return result


# ============================================================
# UI
# ============================================================

st.sidebar.header("Evaluation Inputs")

problem_file = st.sidebar.file_uploader(
    "Problem Statement",
    type=["pdf", "docx", "txt", "md"],
)

rubric_file = st.sidebar.file_uploader(
    "Evaluation Rubric",
    type=["xlsx"],
)

submission_files = st.sidebar.file_uploader(
    "Student Submission(s)",
    type=[
        "ipynb",
        "zip",
        "docx",
        "pdf",
        "html",
        "htm",
        "py",
        "md",
        "txt",
    ],
    accept_multiple_files=True,
)

custom_instructions = st.text_area(
    "Additional Strict Evaluation Instructions",
    value=(
        "Be strict. Award marks only when implementation evidence exists. "
        "Do not award full marks for descriptions alone. "
        "Deduct for missing components, incorrect models, hard-coded implementations, "
        "missing PDF upload functionality, missing grounding, hallucination, poor "
        "error handling, incomplete integration, or missing documentation. "
        "Only genuinely complete and well-tested submissions should score 90+."
    ),
    height=150,
)

if problem_file:
    st.success(f"Problem statement loaded: {problem_file.name}")

if rubric_file:
    st.success(f"Rubric loaded: {rubric_file.name}")

if submission_files:
    st.success(f"{len(submission_files)} submission(s) loaded.")


# ============================================================
# EVALUATE
# ============================================================

if st.button("🚀 Evaluate Submissions", type="primary"):

    if client is None:
        st.error("OpenAI API configuration is missing.")
        st.stop()

    if problem_file is None:
        st.error("Please upload the problem statement PDF/DOCX.")
        st.stop()

    if rubric_file is None:
        st.error("Please upload the evaluation rubric Excel file.")
        st.stop()

    if not submission_files:
        st.error("Please upload at least one student submission.")
        st.stop()

    problem_text = read_problem(problem_file)

    if not problem_text.strip():
        st.warning(
            "Problem statement text could not be extracted. "
            "Evaluation will continue using the rubric."
        )

    try:
        raw_rubric = pd.read_excel(rubric_file)
    except Exception as exc:
        st.error(f"Could not read rubric Excel: {exc}")
        st.stop()

    rubric_df, missing = normalize_rubric(raw_rubric)

    if rubric_df is None or rubric_df.empty:
        st.error(
            "Rubric is invalid. Required columns are: "
            "Criterion, Max Score, Description."
        )
        st.stop()

    if missing:
        st.error(f"Missing rubric fields: {missing}")
        st.stop()

    rubric_text = rubric_to_text(rubric_df)

    with st.expander("Rubric used for evaluation", expanded=False):
        st.dataframe(rubric_df, use_container_width=True)

    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=min(MAX_WORKERS, len(submission_files))
    ) as executor:

        futures = {
            executor.submit(
                process_submission,
                file,
                problem_text,
                rubric_df,
                rubric_text,
                custom_instructions,
            ): file.name
            for file in submission_files
        }

        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            filename = futures[future]

            try:
                results.append(future.result())
                status.info(
                    f"Evaluated {completed}/{len(futures)}: {filename}"
                )
            except Exception as exc:
                logging.exception(
                    "Unexpected evaluation failure for %s",
                    filename,
                )
                results.append({
                    "Participant": filename,
                    "Total": 0,
                    "Max Total": 100,
                    "Percentage": 0,
                    "Rating": "Evaluation Failed",
                    "Overall Feedback": str(exc),
                })

            progress.progress(
                completed / len(futures)
            )

    output = pd.DataFrame(results)

    st.success("Evaluation completed.")

    # Summary table
    summary_cols = [
        "Participant",
        "Total",
        "Max Total",
        "Percentage",
        "Rating",
    ]

    st.subheader("📌 Evaluation Summary")
    st.dataframe(
        output[
            [c for c in summary_cols if c in output.columns]
        ],
        use_container_width=True,
    )

    # Detailed results
    st.subheader("📋 Detailed Evaluation")

    for _, row in output.iterrows():

        with st.expander(
            f"{row.get('Participant', 'Student')} — "
            f"{row.get('Total', 0)}/{row.get('Max Total', 100)} "
            f"({row.get('Percentage', 0)}%)"
        ):

            score_data = []

            for _, rubric_row in rubric_df.iterrows():
                criterion = str(rubric_row["criterion"])
                score_data.append({
                    "Criterion": criterion,
                    "Max Score": rubric_row["max_score"],
                    "Score": row.get(criterion, 0),
                    "Feedback": row.get(
                        f"{criterion} - Feedback",
                        "",
                    ),
                })

            st.dataframe(
                pd.DataFrame(score_data),
                use_container_width=True,
            )

            st.markdown("### Overall Feedback")
            st.write(row.get("Overall Feedback", ""))

            st.markdown("### Strengths")
            st.write(row.get("Strengths", ""))

            st.markdown("### Improvements")
            st.write(row.get("Improvements", ""))

            st.markdown("### Missing Requirements")
            st.write(row.get("Missing Requirements", ""))

            st.markdown("### Evidence Summary")
            st.write(row.get("Evidence Summary", ""))

            if row.get("Parser Warnings"):
                st.warning(row.get("Parser Warnings"))

    # Download Excel
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="xlsxwriter",
    ) as writer:

        output.to_excel(
            writer,
            index=False,
            sheet_name="Evaluation Results",
        )

        detailed_rows = []

        for _, row in output.iterrows():
            for _, rubric_row in rubric_df.iterrows():
                criterion = str(rubric_row["criterion"])

                detailed_rows.append({
                    "Participant": row.get("Participant", ""),
                    "Criterion": criterion,
                    "Max Score": rubric_row["max_score"],
                    "Score": row.get(criterion, 0),
                    "Feedback": row.get(
                        f"{criterion} - Feedback",
                        "",
                    ),
                    "Total": row.get("Total", 0),
                    "Percentage": row.get("Percentage", 0),
                    "Rating": row.get("Rating", ""),
                })

        pd.DataFrame(detailed_rows).to_excel(
            writer,
            index=False,
            sheet_name="Task-wise Scores",
        )

    st.download_button(
        "📥 Download Evaluation Excel",
        data=excel_buffer.getvalue(),
        file_name="PDF_RAG_Evaluation_Report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )
