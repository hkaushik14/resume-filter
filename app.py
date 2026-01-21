import streamlit as st
import pymupdf as fitz
from sentence_transformers import SentenceTransformer, util
import requests
import os
from dotenv import load_dotenv
import re
from spellchecker import SpellChecker
import torch
import asyncio
import spacy
# Setup event loop fix for Windows if needed
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

load_dotenv()

st.set_page_config(page_title="🤖 AI Resume Filter Agent", layout="wide")
st.title("🤖 AI Resume Filter Agent")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

resume_text = None
if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()
    st.subheader("📄 Resume Text Preview")
    st.text(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))

job_description = st.text_area("Paste the Job Description here:",
                               placeholder="Enter the job description text here...")

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop and token.is_alpha and len(token.text) > 3:
            keywords.add(token.text)
    return keywords

def basic_format_check(text):
    issues = []

    # Check for ALL CAPS words (more than 3 letters)
    all_caps = re.findall(r'\b[A-Z]{4,}\b', text)
    if all_caps:
        issues.append(f"Too many ALL CAPS words: {', '.join(set(all_caps))}")

    # Check if contact info (email, phone) present
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\d{10})', text)
    if not email_match:
        issues.append("Email address seems missing or invalid.")
    if not phone_match:
        issues.append("Phone number seems missing or invalid.")

    # Check paragraph length (assuming paragraphs separated by newlines)
    paragraphs = text.split('\n\n')
    long_paras = [p for p in paragraphs if len(p.split()) > 100]
    if long_paras:
        issues.append("Resume contains very long paragraphs. Use bullet points for clarity.")

    # Spell check first 500 words (to save time)
    spell = SpellChecker()
    words = re.findall(r'\b\w+\b', text.lower())[:500]
    misspelled = spell.unknown(words)
    if misspelled:
        # Whitelist common words to avoid false positives
        whitelist = {'gmail', 'html', 'css', 'js', 'coer', 'kaushik', 'harshkaushik494', 'docker', 'tech', 'apps'}
        misspelled = misspelled - whitelist
        if misspelled:
            issues.append(f"Possible spelling mistakes detected: {', '.join(list(misspelled)[:10])}")

    return issues

def find_irrelevant_skills(resume_text, jd_text):
    jd_skills = set(re.findall(r'\b\w{3,}\b', jd_text.lower()))
    resume_skills = set(re.findall(r'\b\w{3,}\b', resume_text.lower()))
    irrelevant = resume_skills - jd_skills
    # Filtering out some general/common words
    common_words = {'and', 'the', 'for', 'with', 'this', 'that', 'from', 'your', 'will', 'have', 'are'}
    irrelevant = irrelevant - common_words
    return list(irrelevant)[:20]

def get_hf_embedding(text):
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise ValueError("Hugging Face API key not set in .env")

    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error: {response.text}")

    embedding = response.json()
    # Convert nested list to tensor and average over tokens dimension
    return torch.tensor(embedding).mean(dim=1)

if st.button("Analyze Resume"):

    if not resume_text or not job_description.strip():
        st.error("Please upload a resume and paste the job description!")
    else:
        with st.spinner("Computing semantic similarity..."):
            try:
                resume_embedding = model.encode(resume_text, convert_to_tensor=True)
                jd_embedding = model.encode(job_description, convert_to_tensor=True)
            except Exception as e:
                st.warning("Local model embedding failed, using Hugging Face API fallback.")
                try:
                    resume_embedding = get_hf_embedding(resume_text)
                    jd_embedding = get_hf_embedding(job_description)
                except Exception as e:
                    st.error(f"Failed to get embeddings: {e}")
                    st.stop()

            cosine_scores = util.pytorch_cos_sim(resume_embedding, jd_embedding)
            score_percent = float(cosine_scores[0][0]) * 100

            st.subheader("📊 Resume vs Job Description Similarity Score")
            st.write(f"Similarity Score: **{score_percent:.2f}%**")

            jd_keywords = extract_keywords(job_description)
            resume_keywords = extract_keywords(resume_text)
            missing_skills = jd_keywords - resume_keywords
            if missing_skills:
                st.subheader("⚠️ Missing Skills / Keywords from Resume")
                st.write(", ".join(sorted(missing_skills)))

            issues = basic_format_check(resume_text)
            if issues:
                st.subheader("⚠️ Formatting / Content Issues Found:")
                for issue in issues:
                    st.write(f"- {issue}")
                st.info("💡 Tip: Use bullet points instead of long paragraphs. Also, consider whitelisting common proper nouns in spellcheck.")
            else:
                st.success("No major formatting or obvious content issues found.")

            irrelevant_skills = find_irrelevant_skills(resume_text, job_description)
            if irrelevant_skills:
                st.subheader("🛑 Skills that may be irrelevant to Job Description:")
                st.write(", ".join(irrelevant_skills))
            else:
                st.success("All skills seem relevant to the job description.")

            # Simple resume summary
use_hf_summary = st.checkbox("Generate Resume Summary using Hugging Face API (optional)")

if use_hf_summary and resume_text:
    with st.spinner("Generating summary with Hugging Face..."):
        HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

        if not HF_API_KEY:
            st.error("Hugging Face API key not found. Please set HUGGINGFACE_API_KEY in your .env file.")
        else:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            payload = {"inputs": resume_text}

            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                summary = response.json()
                summary_text = ""
                if isinstance(summary, list):
                    summary_text = summary[0].get("summary_text", "")
                elif isinstance(summary, dict):
                    summary_text = summary.get("summary_text", "")

                if summary_text:
                    st.subheader("📝 Resume Summary")
                    st.write(summary_text)
                else:
                    st.warning("Could not get summary in expected format from Hugging Face.")
            else:
                st.error(f"Hugging Face API error: {response.status_code} - {response.text}")

