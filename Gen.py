import streamlit as st
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi
import os
import PyPDF2 as pdf
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Set the API key
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GENAI_API_KEY)

# Set page configuration
st.set_page_config(page_title="GenAI", page_icon="🤖", layout="wide")


# Function to load Gemini and get responses
def get_gemini_response(prompt, image=None):
    model = (
        genai.GenerativeModel("gemini-pro-vision")
        if image
        else genai.GenerativeModel("gemini-pro")
    )
    response = model.generate_content(prompt if not image else [prompt, image])
    return response.text
    # Concatenate the parts of the response
    response_text = "".join(
        [part.content for part in response.candidates[0].content.parts]
    )
    return response_text


# Function to extract transcript from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        raise e


# Function to generate summary from transcript using Gemini
def generate_gemini_summary(transcript_text):
    prompt = """You are a YouTube video summarizer. Summarize the entire video and provide the important summary in points within 300 words. Please provide the summary of the text given here: """
    model = genai.GenerativeModel("gemini-pro")  # Load the model within the function
    response = model.generate_content(prompt + transcript_text)
    return response.text


# Function to summarize a long article or blog post using Gemini
def summarize_long_text(long_text):
    prompt = """You are a text summarizer. Summarize the provided text and provide the key points within 250 words."""
    model = genai.GenerativeModel("gemini-pro")  # Load the model within the function
    response = model.generate_content(prompt + long_text)
    return response.text


# Function to extract text from a PDF file
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text


# Prompt Template
input_prompt = """
Hey Act Like a skilled or very experienced ATS (Application Tracking System)
with a deep understanding of the tech field, software engineering, Cloud engineering, DevOps engineering, data science, data analyst,
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive, and you should provide 
the best assistance for improving the resumes. Assign the percentage Matching based 
on JD and
the missing keywords with high accuracy
resume:{text}
description:{jd}

I want the response in one single string having the structure
{{"JD Match":"%",


"MissingKeywords": [],


"Profile Summary": ""}}
"""

# ... (other existing functions)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


def display_ats_section():
    st.subheader("Improve Your Resume Using ATS")
    job_description = st.text_area(
        "Job Description",
        help="Provide a detailed job description for evaluation.",
        placeholder="Enter the job description here...",
        height=200,
    )
    uploaded_file = st.file_uploader(
        "Upload Your Resume (PDF)",
        type="pdf",
        help="Upload your resume in PDF format for evaluation.",
        accept_multiple_files=False,
    )
    submit = st.button("Evaluate ATS", use_container_width=True)

    if submit:
        if uploaded_file is not None:
            with st.spinner("Evaluating resume..."):
                text = input_pdf_text(uploaded_file)
                response = get_gemini_response(
                    input_prompt.format(text=text, jd=job_description)
                )
            st.success("ATS evaluation result:")
            st.write(response)
        else:
            st.error("Please upload your resume (PDF format) to proceed.")


def main():
    def format_func(option):
        icons = {
            "Gen-AI Chatbot": "🤖",
            "Gen-AI Image-based Q&A": "🖼️",
            "Gen-AI YouTube Video Summarizer": "📺",
            "Gen-AI Text Summarizer": "📄",
            "Gen-AI Smart ATS": "📋",
            "Gen-AI Chat with PDF": "📂",
        }
        return f"{icons.get(option, '')} {option}"

    # Sidebar navigation
    with st.sidebar:
        st.title("GenAI")
        nav_options = [
            "Gen-AI Chatbot",
            "Gen-AI Image-based Q&A",
            "Gen-AI YouTube Video Summarizer",
            "Gen-AI Text Summarizer",
            "Gen-AI Smart ATS",
            "Gen-AI Chat with PDF",
        ]
        selected_page = st.radio("Navigation", nav_options, format_func=format_func)

    # Main content
    with st.container():
        st.title("GenAI Multifunctional App")

        if selected_page == "Gen-AI Chatbot":
            input_text = st.text_input(
                "Ask me anything:",
                help="Enter your question for the chatbot.",
                placeholder="Type your question here...",
                key="chatbot_input",
            )
            submit_button = st.button("Get Answer", use_container_width=True)

            if submit_button:
                with st.spinner("Generating response..."):
                    response = get_gemini_response(input_text)
                st.success("Response:")
                st.write(response)

        elif selected_page == "Gen-AI Image-based Q&A":
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                help="Upload an image for analysis.",
                disabled=False,
                label_visibility="visible",
            )
            image_prompt = st.text_input(
                "Question about the image:",
                help="Ask a question related to the uploaded image.",
                placeholder="Type your question here...",
                key="image_input",
            )
            submit_image_button = st.button(
                "Tell me about the image", use_container_width=True
            )

            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

            if submit_image_button:
                with st.spinner("Analyzing image..."):
                    response = get_gemini_response(image_prompt, image)
                st.success("Response:")
                st.write(response)

        elif selected_page == "Gen-AI YouTube Video Summarizer":
            youtube_link = st.text_input(
                "Enter YouTube video link:",
                help="Provide the link to the YouTube video for summarization.",
                placeholder="Enter YouTube video link...",
                key="youtube_input",
            )
            summarize_button = st.button("Summarize video", use_container_width=True)

            if youtube_link:
                video_id = youtube_link.split("=")[1]
                st.image(
                    f"http://img.youtube.com/vi/{video_id}/0.jpg",
                    use_column_width=True,
                    caption="Video Thumbnail",
                )

            if summarize_button:
                with st.spinner("Generating summary..."):
                    transcript_text = extract_transcript_details(youtube_link)
                    if transcript_text:
                        summary = generate_gemini_summary(transcript_text)
                st.success("Video Summary:")
                st.write(summary)

        elif selected_page == "Gen-AI Text Summarizer":
            long_text = st.text_area(
                "Paste your long article or blog post here:",
                help="Paste the text you want to summarize.",
                placeholder="Paste your text here...",
                height=300,
                key="text_input",
            )
            summarize_text_button = st.button(
                "Summarize Text", use_container_width=True
            )

            if summarize_text_button:
                with st.spinner("Generating summary..."):
                    summary = summarize_long_text(long_text)
                st.success("Text Summary:")
                st.write(summary)

        elif selected_page == "Gen-AI Smart ATS":
            display_ats_section()

        elif selected_page == "Gen-AI Chat with PDF":
            pdf_docs = st.file_uploader(
                "Upload your PDF files",
                accept_multiple_files=True,
                help="Upload PDF files for chatbot interaction.",
                disabled=False,
                label_visibility="visible",
            )
            if st.button("Process PDFs", use_container_width=True):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")

            st.subheader("Ask a Question from the PDF Files")
            user_question = st.text_input(
                "",
                help="Ask a question related to the content of the uploaded PDF files.",
                placeholder="Type your question here...",
                key="pdf_input",
            )
            if user_question:
                with st.spinner("Generating response..."):
                    user_input(user_question)


if __name__ == "__main__":
    main()
