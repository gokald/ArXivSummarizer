import os
import re
import warnings
import wget
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

warnings.filterwarnings("ignore")


def extract_arxiv_id(arxiv_url: str) -> str:
    """Extracts arXiv ID from URL or returns input if not found."""
    match = re.search(r"arxiv\.org/abs/(\d+\.\d+)|(\d+\.\d+)", arxiv_url)
    if match:
        return match.group(1) or match.group(2)
    return arxiv_url.strip()


def download_arxiv_pdf(arxiv_url: str, out_filename: str = "paper.pdf") -> str:
    """Downloads PDF from arXiv URL, returns local filename."""
    arxiv_id = extract_arxiv_id(arxiv_url)
    pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        print(f"Downloading {pdf_link}...")
        wget.download(pdf_link, out=out_filename)
        print(f"\nSaved as {out_filename}")
    except Exception as e:
        print(f"Download failed: {e}")
        raise
    
    return out_filename


def pdf_to_vectorstore(pdf_file: str, chunk_size=800, chunk_overlap=100):
    """Processes PDF into vector embeddings for search."""
    loader = PyPDFLoader(pdf_file)
    
    try:
        pages = loader.load_and_split()
    except Exception as e:
        print(f"PDF processing error: {e}")
        raise

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    docs = splitter.split_documents(pages)

    print(f"Creating vector store with {len(docs)} chunks...")
    return Chroma.from_documents(docs, HuggingFaceEmbeddings())


def build_watsonx_llm(
    model_id: str,
    decoding_method=DecodingMethods.GREEDY,
    max_tokens=256,
    temp=0.5,
    min_tokens=50,
    credentials=None,
    project_id=None # replace with your ibmcloud project id
):
    """Initializes Watsonx LLM with specified parameters."""
    credentials = credentials or {
        "url": "https://us-south.ml.cloud.ibm.com",
        # "apikey": insert your api key here
    }

    params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.TEMPERATURE: temp,
        GenParams.MIN_NEW_TOKENS: min_tokens
    }

    model = Model(
        model_id=model_id,
        params=params,
        credentials=credentials,
        project_id=project_id
    )
    return WatsonxLLM(model=model)


def ask_question(docsearch, llm, question: str, return_sources=False):
    """Answers questions using document content."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=return_sources
    )
    result = qa_chain({"query": question})
    return result if return_sources else result["result"]


def safe_qa_prompt(docsearch, llm, question: str):
    """Answers with 'I don't know' when information is missing."""
    prompt = """Use the document to answer the question. If unsure, say 'I don't know'.

{context}

Question: {question}

Answer:"""

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": PromptTemplate(
            template=prompt, 
            input_variables=["context", "question"]
        )}
    )
    return qa_chain({"query": question})["result"]


def summarize_with_rag(docsearch, model_id="ibm/granite-13b-chat-v2"):
    """Generates paper summary using RAG approach."""
    print("\nGenerating summary...")
    llm = build_watsonx_llm(model_id=model_id)
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=docsearch.as_retriever()
    )

    return rag_chain({"query": "Summarize key results and contributions:"})["result"]


def main():
    """Main workflow for arXiv paper analysis."""
    arxiv_url = input("Enter arXiv URL: ").strip()
    if not arxiv_url:
        print("No URL provided")
        return

    pdf_file = download_arxiv_pdf(arxiv_url, "paper.pdf")
    vector_store = pdf_to_vectorstore(pdf_file)

    print("\n==== Summary ====")
    print(summarize_with_rag(vector_store))
    print("=================\n")

    qa_llm = build_watsonx_llm(model_id="meta-llama/llama-3-1-8b-instruct")
    
    print("Ask questions about the paper (type 'quit' to exit):")
    while True:
        query = input("\nYour question: ")
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
            
        answer = safe_qa_prompt(vector_store, qa_llm, query)
        print(f"\n{'-'*50}\n{query}\n{'-'*50}\n{answer}\n")


if __name__ == "__main__":
    main()