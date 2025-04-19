import streamlit as st
import openai
from sentence_transformers import SentenceTransformer, util
import faiss
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import textwrap
from textwrap import wrap
import re
import fitz
from io import BytesIO
import nltk
import pandas as pd
from serpapi import GoogleSearch

def web_search(query, num_results=3):
    params = {
        "engine": "google",
        "q": query,
        "api_key": "API",  # 🔑 Replace with your actual SerpAPI key
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = []
    for result in results.get("organic_results", []):
        if "snippet" in result:
            snippets.append(result["snippet"])
    return " ".join(snippets)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up your OpenAI API key
openai.api_key = "API"

# Define a default plan that can be edited by the user
default_plan = """
1. Analyze the question.
2. Reference the uploaded PDF document.
3. Use the provided text to gsenerate a detailed response. 
4. Answer in 4 lines
"""

st.title('PDF Text Manipulation Tool')

def extract_headings(pdf_path):
    """Extract unique headings from a PDF that start with a number, are in uppercase, may contain special characters but do not end with a special character."""
    doc = fitz.open(pdf_path)
    headings = set()  # Use a set to avoid duplicates
    pattern = re.compile(r'^\d+\s+[A-Z\s/()-]+[A-Z0-9]$')

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block['type'] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span['text'].strip()
                        if pattern.match(text):
                            headings.add(text)  # Add to set, automatically avoiding duplicates

    # Convert set to a sorted list
    headings_sorted = sorted(list(headings), key=lambda x: int(x.split()[0]))
    return headings_sorted

def extract_sections(doc, start_section, end_section_pattern):
    """Extract sections from a PDF based on the start_section and end_section_pattern."""
    all_sections_text = []
    current_section_text = []
    is_extracting = False
    for page in doc:
        text = page.get_text("text")
        for line in text.splitlines():
            if start_section in line and not is_extracting:
                is_extracting = True
                current_section_text = [line]
                continue
            elif is_extracting and re.match(end_section_pattern, line):
                all_sections_text.append("\n".join(current_section_text))
                current_section_text = []
                is_extracting = False
                break
            if is_extracting:
                current_section_text.append(line)
        if is_extracting:
            all_sections_text.append("\n".join(current_section_text))
    return "\n\n".join(all_sections_text)

def save_text_to_pdf(text):
    """Save text to a PDF file."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text_object = c.beginText(40, 750)
    text_object.setFont("Helvetica", 12)
    for line in text.split('\n'):
        wrapped_lines = textwrap.wrap(line, width=100)
        for wrapped_line in wrapped_lines:
            text_object.textLine(wrapped_line)
            if text_object.getY() <= 72:
                c.drawText(text_object)
                c.showPage()
                text_object = c.beginText(40, 750)
    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
if uploaded_file:
    temp_pdf_path = "temporary_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    doc = fitz.open(temp_pdf_path)
    headings = extract_headings(temp_pdf_path)
    if headings:
        selected_heading = st.selectbox("Select a section heading:", headings)
        editable_heading = st.text_input("Edit or confirm the selected heading:", value=selected_heading)

        if st.button("Extract Text Using Edited Heading"):
            extracted_text = extract_sections(doc, editable_heading, r"^\d+[\s.]*[A-Z ]+$")
            st.session_state['extracted_text'] = extracted_text  # Store extracted text in session state

    if 'extracted_text' in st.session_state:
        edited_text = st.text_area("Edit the extracted text if needed:", st.session_state['extracted_text'], height=300)
        if st.button("Finalize Text"):
            final_text = edited_text  # Use the edited text for further processing
            st.write("Final text processed!")
            result_pdf = save_text_to_pdf(final_text)
            st.download_button("Download Final Text as PDF", result_pdf, file_name="final_text.pdf")
            result_text = final_text.encode('utf-8')
            st.download_button("Download Final Text as Text", result_text, file_name="final_text.txt")


# Streamlit App
st.title("Document-Based AI Chatbot")
st.write("Upload documents and ask questions based on selected documents!")

# Before the chat input section:
st.subheader("Chat")
st.text_area("Modify Scratchpad Plan:", value=default_plan, height=300, key="scratchpad_plan")  # Allows the user to modify how they approach answering questions
if "scratchpad_plan" not in st.session_state:
    st.session_state["scratchpad_plan"] = default_plan
scratchpad_plan = st.session_state["scratchpad_plan"]
# Initialize session state
if "package_texts" not in st.session_state:
    st.session_state.package_texts = {}
if "reference_texts" not in st.session_state:
    st.session_state.reference_texts = {}
if "package_index" not in st.session_state:
    st.session_state.package_index = {}
if "reference_index" not in st.session_state:
    st.session_state.reference_index = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Predefined questions
predefined_questions = [
    "Does the _ package insert align with the _ reference documents for hypoglycemia management?",
    "Does the _ package insert adequately address the risk management for patients with a history of hemorrhagic stroke as outlined in the _ FDA guidance?",
    "How does the _ labeling address the administration of the drug in pregnant women as compared to the recommendations provided in the _ FDA guidance document?.",
    "How does the _ labeling manage the recommendation against breastfeeding in the _ section compared to the FDA's guidelines?",
     "Does _ effectively manage depression and anxiety in pregnant women without significant risks to the mother and fetus, as outlined in the _ FDA guidelines for drug use during pregnancy"
]

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to save chat history to PDF
def save_chat_to_pdf(chat_history, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    text_object = c.beginText(40, 750)
    text_object.setFont("Helvetica", 10)
    max_y = 40

    for chat in chat_history:
        question = f"Question: {chat['user']}"
        answer = f"Answer: {chat['assistant']}"
        for line in [question, answer, ""]:
            wrapped_lines = wrap(line, width=90)
            for wrapped_line in wrapped_lines:
                if text_object.getY() <= max_y:
                    c.drawText(text_object)
                    c.showPage()
                    text_object = c.beginText(40, 750)
                    text_object.setFont("Helvetica", 10)
                text_object.textLine(wrapped_line)
    c.drawText(text_object)
    c.save()
    return output_path

# File Upload
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Package Inserts")
    package_files = st.file_uploader("Upload PDF files for Package Inserts", type=["pdf"], accept_multiple_files=True, key="package_files")

    if package_files:
        for uploaded_file in package_files:
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.package_texts[uploaded_file.name] = text
            sentences = text.split(". ")
            embeddings = model.encode(sentences, convert_to_tensor=False)
            st.session_state.package_index[uploaded_file.name] = create_faiss_index(embeddings)
        st.write("Package inserts processed successfully!")

with col2:
    st.subheader("Upload Reference Documents")
    reference_files = st.file_uploader("Upload PDF files for Reference Documents", type=["pdf"], accept_multiple_files=True, key="reference_files")

    if reference_files:
        for uploaded_file in reference_files:
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.reference_texts[uploaded_file.name] = text
            sentences = text.split(". ")
            embeddings = model.encode(sentences, convert_to_tensor=False)
            st.session_state.reference_index[uploaded_file.name] = create_faiss_index(embeddings)
        st.write("Reference documents processed successfully!")

# Select documents for querying
st.subheader("Select Documents for Querying")
package_docs_to_search = st.multiselect(
    "Select Package Inserts to Search From:",
    list(st.session_state.package_texts.keys())
)
reference_docs_to_search = st.multiselect(
    "Select Reference Documents to Search From:",
    list(st.session_state.reference_texts.keys())
)

# Chat Input
st.subheader("Chat")
# Dropdown for predefined questions
question = st.selectbox("Choose a predefined question or type your own below:", predefined_questions)
user_query = st.text_input("Or type your question here:", key="query_input")
query = user_query if user_query else question

use_web = st.checkbox("Include Web Search (Live Internet Results)")
st.session_state["use_web"] = use_web  # Save to session if needed elsewhere

# --- Submit button and processing logic ---
if st.button("Submit"):
    scratchpad_plan = st.session_state.scratchpad_plan
    selected_texts = ""
    selected_indices = []

    # Use selected package inserts
    if package_docs_to_search:
        for doc in package_docs_to_search:
            selected_texts += st.session_state.package_texts[doc]
            selected_indices.append(st.session_state.package_index[doc])

    # Use selected reference documents
    if reference_docs_to_search:
        for doc in reference_docs_to_search:
            selected_texts += st.session_state.reference_texts[doc]
            selected_indices.append(st.session_state.reference_index[doc])

    # Default to all documents if no selection is made
    if not package_docs_to_search and not reference_docs_to_search:
        selected_texts = " ".join(st.session_state.package_texts.values()) + " " + " ".join(st.session_state.reference_texts.values())
        selected_indices = list(st.session_state.package_index.values()) + list(st.session_state.reference_index.values())

    # Retrieve relevant content using FAISS
    sentences = selected_texts.split(". ")
    embeddings = model.encode(sentences, convert_to_tensor=False)
    index = create_faiss_index(embeddings)
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, k=3)
    relevant_sentences = [sentences[i] for i in indices[0]]

    # Combine relevant sentences for context
    context = " ".join(relevant_sentences)
    if not st.session_state.get("use_web", False) and not context.strip():
        st.session_state.chat_history.append({
            "user": query,
            "assistant": "I'm unable to find relevant information in the uploaded documents to answer your question."
        })
    else:
        # Add web search context if enabled
        if use_web:
            try:
                web_context = web_search(query)
                context += f"\n\n[Web Search Info]:\n{web_context}"
            except Exception as e:
                st.warning(f"Web search failed: {e}")

        # OpenAI GPT API Integration
        prompt = f"""
You are a document-restricted assistant.

Instructions:
1. You are only allowed to answer using the content from the uploaded documents provided in the context below.
2. If the answer is not directly supported by the context, reply: "This information is not available in the uploaded documents."
3. Do not rely on your internal knowledge. Do not fabricate information.
{scratchpad_plan}
Context: {context}
Question: {query}
Provide a detailed and accurate answer.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            answer = response['choices'][0]['message']['content'].strip()
            st.session_state.chat_history.append({"user": query, "assistant": answer})
        except Exception as e:
            st.error(f"Error generating answer: {e}")

# --- Always show Chat History ---
st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.write(f"**Question:** {chat['user']}")
    st.write(f"**Answer:** {chat['assistant']}")

if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")


   


# Download Button for Chat History
if st.session_state.chat_history:
    st.subheader("Download Chat History")
    output_path = "chat_history.pdf"
    file_path = save_chat_to_pdf(st.session_state.chat_history, output_path)
    with open(file_path, "rb") as file:
        btn = st.download_button(
            label="Download Chat History as PDF",
            data=file,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )
        if btn:
            st.success("Download started!")
