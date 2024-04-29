import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Initialize Pinecone with your API key
pinecone_api_key = "aa9da95a-d147-48a1-bfec-e78bea838890"
pc = PineconeClient(api_key=pinecone_api_key)

# Create a new Pinecone index
index_name = "my-pdf-index"  # Use a valid index name
dimension = 1536  # Dimension of the embeddings
metric = "cosine"  # Similarity metric for the index

@st.cache_resource
def load_pdf_and_create_index():
    # Load the PDF document
    loader = PyPDFLoader("AMM_19_09_16_Rev32_TRs_incorporated (2).pdf")
    documents = loader.load()

    # Split the text into larger chunks
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Specify the region as 'us-east-1'
            )
        )

    # Create embeddings for the text chunks
    embeddings = OpenAIEmbeddings()

    # Create the Pinecone index
    docsearch = PineconeVectorStore(
        index_name=index_name,
        namespace="my_namespace",
        embedding=embeddings,
        text_key="text",
        pinecone_api_key=pinecone_api_key
    )

    # Add section information to the metadata
    metadatas = [{"source": t.metadata["source"], "section": f"Section {i+1}"} for i, t in enumerate(texts)]
    docsearch.add_texts([t.page_content for t in texts], metadatas=metadatas)

    return docsearch

# Load the PDF and create the index
docsearch = load_pdf_and_create_index()

# Initialize the language model (GPT-4)
llm = ChatOpenAI(model_name="gpt-4")

# Define a prompt template that includes the source information
prompt_template = """
Based on the provided context, answer the question as thoroughly as possible. 
Also, include the source PDF and section for each part of your answer.

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the retrieval-augmented QA chain with the custom prompt
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
)

# Streamlit app
def main():
    st.title("PDF Question Answering App")
    query = st.text_input("Enter your question:")

    if st.button("Ask"):
        if query:
            # Cache the question-answering process
            result = ask_question(query)
            st.write(result)
        else:
            st.write("Please enter a question.")

@st.cache_data
def ask_question(query):
    return qa.invoke(query)

if __name__ == "__main__":
    main()
