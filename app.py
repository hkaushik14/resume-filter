import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz
import google.generativeai as genai
import re
from spellchecker import SpellChecker
import torch
import spacy

# Configure Gemini API from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def load_gemini():
    return genai.GenerativeModel("gemini-1.5-flash")

gemini_model = load_gemini()

# Streamlit page setup
st.set_page_config(page_title="🤖 AI Resume Filter Agent", layout="wide")
st.title("🤖 AI Resume Filter Agent")

# Load SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

resume_text = None
if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()
    st.subheader("📄 Resume Text Preview")
    st.text(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))

# Job description input
job_description = st.text_area(
    "Paste the Job Description here:",
    placeholder="Enter the job description text here..."
)

# Load spaCy model
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# Extract keywords from text
def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop and token.is_alpha and len(token.text) > 3:
            keywords.add(token.text)
    return keywords

# Basic format and content checks
def basic_format_check(text):
    issues = []

    # ALL CAPS words
    all_caps = re.findall(r'\b[A-Z]{4,}\b', text)
    if all_caps:
        issues.append(f"Too many ALL CAPS words: {', '.join(set(all_caps))}")

    # Email and phone check
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\d{10})', text)
    if not email_match:
        issues.append("Email address seems missing or invalid.")
    if not phone_match:
        issues.append("Phone number seems missing or invalid.")

    # Paragraph length
    paragraphs = text.split('\n\n')
    long_paras = [p for p in paragraphs if len(p.split()) > 100]
    if long_paras:
        issues.append("Resume contains very long paragraphs. Use bullet points for clarity.")

    # Spell check first 500 words
    spell = SpellChecker()
    words = re.findall(r'\b\w+\b', text.lower())[:500]
    misspelled = spell.unknown(words)
    if misspelled:
        whitelist = {'gmail', 'html', 'css', 'js', 'coer', 'kaushik', 'harshkaushik494', 'docker', 'tech', 'apps'}
        misspelled = misspelled - whitelist
        if misspelled:
            issues.append(f"Possible spelling mistakes detected: {', '.join(list(misspelled)[:10])}")

    return issues

# Find irrelevant skills
def find_irrelevant_skills(resume_text, jd_text):
    jd_skills = set(re.findall(r'\b\w{3,}\b', jd_text.lower()))
    resume_skills = set(re.findall(r'\b\w{3,}\b', resume_text.lower()))
    irrelevant = resume_skills - jd_skills
    common_words = {'and', 'the', 'for', 'with', 'this', 'that', 'from', 'your', 'will', 'have', 'are'}
    irrelevant = irrelevant - common_words
    return list(irrelevant)[:20]

# Analyze resume button
if st.button("Analyze Resume"):

    if not resume_text or not job_description.strip():
        st.error("Please upload a resume and paste the job description!")
    else:
        with st.spinner("Computing semantic similarity..."):
            # Encode embeddings
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            jd_embedding = model.encode(job_description, convert_to_tensor=True)

            cosine_scores = util.pytorch_cos_sim(resume_embedding, jd_embedding)
            score_percent = float(cosine_scores[0][0]) * 100

            st.subheader("📊 Resume vs Job Description Similarity Score")
            st.write(f"Similarity Score: **{score_percent:.2f}%**")

            # Missing skills / keywords
            jd_keywords = extract_keywords(job_description)
            resume_keywords = extract_keywords(resume_text)
            missing_skills = jd_keywords - resume_keywords
            if missing_skills:
                st.subheader("⚠️ Missing Skills / Keywords from Resume")
                st.write(", ".join(sorted(missing_skills)))

            # Formatting / content issues
            issues = basic_format_check(resume_text)
            if issues:
                st.subheader("⚠️ Formatting / Content Issues Found:")
                for issue in issues:
                    st.write(f"- {issue}")
                st.info("💡 Tip: Use bullet points instead of long paragraphs. Also, consider whitelisting common proper nouns in spellcheck.")
            else:
                st.success("No major formatting or obvious content issues found.")

            # Irrelevant skills
            irrelevant_skills = find_irrelevant_skills(resume_text, job_description)
            if irrelevant_skills:
                st.subheader("🛑 Skills that may be irrelevant to Job Description:")
                st.write(", ".join(irrelevant_skills))
            else:
                st.success("All skills seem relevant to the job description.")

# Gemini resume summary
use_gemini_summary = st.checkbox("Generate Resume Summary using Gemini")

if use_gemini_summary and resume_text:
    with st.spinner("Generating summary..."):
        prompt = f"""
        Summarize this resume in 4-5 bullet points.
        Focus on skills, experience, and impact.

        Resume:
        {resume_text[:12000]}
        """
        response = gemini_model.generate_content(prompt)
        st.subheader("📝 Resume Summary")
        st.write(response.text)
