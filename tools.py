"""
tools.py
--------
Yaha 4 wota CrewAI tools define gareko chu:

  1. college_admission_search   -> RAG over college admission data
  2. it_career_search           -> RAG over IT career/salary/future data
  3. web_search                 -> DuckDuckGo live web search (no API key needed)
  4. optimize_uploaded_cv       -> reads the most-recently-uploaded CV from disk,
                                    optimizes it via Groq, saves .docx, returns summary

Ekai agent le query heryara yi 4 ma bata sahi tool(s) choose garcha.
"""

import os
from crewai.tools import tool
from ddgs import DDGS

from rag_engine import get_college_rag, get_career_rag
from cv_processor import extract_text_from_cv, optimize_cv_text, save_cv_as_docx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Path to whatever CV was most recently uploaded via the Streamlit sidebar.
# app.py writes this file; the tool below just reads it.
LATEST_CV_POINTER = os.path.join(UPLOADS_DIR, "_latest_cv_path.txt")


@tool("College Admission Knowledge Base Search")
def college_admission_search(query: str) -> str:
    """
    Searches a knowledge base about college/university admission process,
    eligibility, entrance exams, required documents, program types
    (BSc.CSIT, BIT, BCA, BE Computer), fees, and scholarships.
    Use this for ANY question about admission requirements, application steps,
    documents needed, entrance exams, or choosing a college/program.
    """
    rag = get_college_rag()
    return rag.query(query, top_k=4)


@tool("IT Career and Salary Knowledge Base Search")
def it_career_search(query: str) -> str:
    """
    Searches a knowledge base about IT career paths (software development,
    data science/AI/ML, cloud/DevOps, cybersecurity, QA, product management),
    salary ranges (Nepal and global), and future outlook/trends in the tech
    industry. Use this for ANY question about career options, which field to
    choose, expected salary, or future scope of an IT career.
    """
    rag = get_career_rag()
    return rag.query(query, top_k=4)


@tool("Live Web Search")
def web_search(query: str) -> str:
    """
    Performs a real-time web search (DuckDuckGo) for current information not
    available in the internal knowledge bases — e.g. latest news, a specific
    company's current job openings, today's exchange rates, very recent
    salary/market data, or specific college's current admission notice.
    Use this when the internal knowledge bases don't have a good answer, or
    the user explicitly asks for the "latest" / "current" / "today's" info.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No web results found."
        formatted = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            formatted.append(f"- {title}\n  {body}\n  Source: {href}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {e}"


@tool("Optimize Uploaded CV")
def optimize_uploaded_cv(target_role: str = "") -> str:
    """
    Reads the CV file the user most recently uploaded (via the sidebar
    uploader), extracts its text, sends it to the Groq LLM to produce a
    refined, ATS-friendly, achievement-oriented version, and saves the
    result as a .docx file in the outputs folder. Pass an optional
    target_role string (e.g. "Backend Developer") if the user mentioned
    what role/job they are optimizing the CV for; otherwise pass an empty
    string. Use this ONLY when the user asks to optimize, refine, improve,
    or review their uploaded CV/resume.
    """
    if not os.path.exists(LATEST_CV_POINTER):
        return ("No CV has been uploaded yet. Please ask the user to upload their "
                "CV file (PDF/DOCX/TXT) using the sidebar uploader first.")

    with open(LATEST_CV_POINTER, "r", encoding="utf-8") as f:
        cv_path = f.read().strip()

    if not os.path.exists(cv_path):
        return "The previously uploaded CV file could not be found. Please re-upload it."

    try:
        raw_text = extract_text_from_cv(cv_path)
        if not raw_text.strip():
            return "Could not extract any text from the uploaded CV. The file might be a scanned image without selectable text."

        optimized_markdown = optimize_cv_text(raw_text, target_role=target_role)

        output_path = os.path.join(OUTPUTS_DIR, "optimized_cv.docx")
        save_cv_as_docx(optimized_markdown, output_path)

        return (
            "CV optimized successfully. Here is the refined CV content:\n\n"
            f"{optimized_markdown}\n\n"
            f"[The optimized CV has also been saved as a downloadable file: {output_path}]"
        )
    except Exception as e:
        return f"CV optimization failed: {e}"


ALL_TOOLS = [college_admission_search, it_career_search, web_search, optimize_uploaded_cv]