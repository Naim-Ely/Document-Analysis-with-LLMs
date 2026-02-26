import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download("punkt")
nltk.download("punkt_tab")

# ==============================
# CONFIG
# ==============================
PDF_PATH = "test2.pdf"   # Change this to your PDF file
MAX_SUMMARY_INPUT = 1000         # Characters for summarization
PASSAGE_WORD_LIMIT = 200         # Words per chunk
MIN_QUESTIONS = 3                # Questions per passage

# ==============================
# DOWNLOAD NLTK DATA
# ==============================
nltk.download("punkt")

# ==============================
# STEP 1: Extract Text from PDF
# ==============================
def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text


# ==============================
# STEP 2: Summarize Document
# ==============================
def summarize_text(text):
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(
        text[:MAX_SUMMARY_INPUT],
        max_length=150,
        min_length=30,
        do_sample=False
    )
    return summary[0]["summary_text"]


# ==============================
# STEP 3: Split Text into Passages
# ==============================
def split_into_passages(text, word_limit=200):
    sentences = sent_tokenize(text)
    passages = []
    current_passage = ""

    for sentence in sentences:
        if len(current_passage.split()) + len(sentence.split()) < word_limit:
            current_passage += " " + sentence
        else:
            passages.append(current_passage.strip())
            current_passage = sentence

    if current_passage:
        passages.append(current_passage.strip())

    return passages


# ==============================
# STEP 4: Generate Questions
# ==============================
def generate_questions(passage, qg_pipeline, min_questions=3):
    input_text = f"generate questions: {passage}"
    result = qg_pipeline(input_text)

    questions = result[0]["generated_text"].split("<sep>")
    questions = [q.strip() for q in questions if q.strip()]

    return questions[:min_questions]


# ==============================
# STEP 5: Answer Questions
# ==============================
def answer_questions(passages, qg_pipeline, qa_pipeline):
    answered = set()

    for idx, passage in enumerate(passages):
        print(f"\n{'='*60}")
        print(f"PASSAGE {idx+1}")
        print(f"{'='*60}\n")
        print(passage[:500], "...\n")

        questions = generate_questions(passage, qg_pipeline, MIN_QUESTIONS)

        for question in questions:
            if question not in answered:
                answer = qa_pipeline({
                    "question": question,
                    "context": passage
                })

                print(f"Q: {question}")
                print(f"A: {answer['answer']}\n")
                answered.add(question)


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    print("Extracting text from PDF...")
    document_text = extract_text_from_pdf(PDF_PATH)

    if not document_text.strip():
        print("No text extracted. Check your PDF.")
        return

    print("\nGenerating Summary...\n")
    summary = summarize_text(document_text)
    print("SUMMARY:")
    print(summary)

    print("\nSplitting document into passages...")
    passages = split_into_passages(document_text, PASSAGE_WORD_LIMIT)

    print("\nLoading Question Generation model...")
    qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

    print("Loading Question Answering model...")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    print("\nGenerating and Answering Questions...\n")
    answer_questions(passages, qg_pipeline, qa_pipeline)


if __name__ == "__main__":
    main()
