import re
from io import BytesIO

import streamlit as st # pyright: ignore[reportMissingImports]
from PyPDF2 import PdfReader # pyright: ignore[reportMissingImports]
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

SKILLS = [
    "python", "java", "c++", "c", "javascript", "typescript", "html", "css",
    "react", "angular", "node.js", "django", "flask", "fastapi", "streamlit",
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "git", "github",
    "docker", "kubernetes", "aws", "azure", "linux", "rest api", "api",
    "machine learning", "deep learning", "nlp", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "data analysis", "excel",
    "power bi", "tableau", "communication", "problem solving"
]

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF held in memory; no file is saved to disk."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()

def find_skills(text: str) -> list[str]:
    normalized = text.lower()
    found = []
    for skill in SKILLS:
        # Match whole terms where possible; escape punctuation such as C++.
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return found

def similarity_score(resume_text: str, job_text: str) -> float:
    """TF-IDF cosine similarity: a useful baseline, not an AI hiring decision."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([resume_text, job_text])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)

st.title("📄 AI Resume Analyzer")
st.caption("Compare a PDF resume with a job description and find skills to improve.")

with st.sidebar:
    st.header("About this project")
    st.write(
        "This demo extracts text from a PDF, identifies skills from a curated "
        "keyword list, and calculates TF-IDF text similarity."
    )
    st.info(
        "This is a learning project, not a validated hiring or ATS system. "
        "Scanned/image-only PDFs may need OCR, which is not included."
    )

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder="Paste the responsibilities, qualifications and required skills here..."
)

if st.button("Analyze resume", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a PDF resume.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            resume_text = extract_pdf_text(uploaded_file.getvalue())
            if len(resume_text) < 30:
                st.error(
                    "Not enough selectable text was found in this PDF. "
                    "Try a text-based PDF rather than a scanned image."
                )
            else:
                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)
                matched = sorted(set(resume_skills) & set(job_skills))
                missing = sorted(set(job_skills) - set(resume_skills))
                score = similarity_score(resume_text, job_description)

                st.subheader("Analysis results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Text similarity", f"{score:.1f}%")
                c2.metric("Matching skills", len(matched))
                c3.metric("Potential missing skills", len(missing))

                st.progress(min(max(int(round(score)), 0), 100))
                left, right = st.columns(2)
                with left:
                    st.markdown("#### ✅ Matching skills")
                    if matched:
                        st.write(", ".join(matched))
                    else:
                        st.write("No skills from the built-in list matched.")
                with right:
                    st.markdown("#### 📌 Skills to review")
                    if missing:
                        st.write(", ".join(missing))
                    else:
                        st.write("No missing skills from the built-in list.")

                report = (
                    "AI RESUME ANALYZER REPORT\n"
                    "========================\n"
                    f"Resume file: {uploaded_file.name}\n"
                    f"Text similarity: {score:.1f}%\n"
                    f"Matching skills: {', '.join(matched) if matched else 'None detected'}\n"
                    f"Potential missing skills: {', '.join(missing) if missing else 'None detected'}\n\n"
                    "Note: This is an educational keyword and text-similarity estimate, "
                    "not a validated ATS score or hiring recommendation.\n"
                )
                st.download_button(
                    "Download analysis report",
                    data=report,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                st.caption(
                    "Privacy: the app processes the uploaded PDF in memory during this "
                    "session and does not intentionally save it to disk."
                )
        except Exception as exc:
            st.error(f"Could not analyze this PDF: {exc}")
