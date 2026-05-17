import asyncio
import os
import ssl
from typing  import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# # Configure SSL context to use certifi's CA bundle
# ssl_context = ssl.create_default_context(cafile=certifi.where())
# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_pages=1000, max_breadth=20)
tavily_crawl = TavilyCrawl()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10)
vector_store = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

async def index_documents_async(documents: List[Document], batch_size: int = 10):
    """Asynchronously index documents in batches."""
    log_header("VECTOR STORE INDEXING")
    log_info(f" Starting indexing of {len(documents)} documents in batches of {batch_size}", Colors.YELLOW)

    # Create Batches of documents
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f" VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents")

    # Process all batches asynchronously
    async def add_batch(batch:List[Document], batch_num: int):
        try:
            await vector_store.add_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} with {len(batch)} documents")
            return True
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num}/{len(batches)} - {str(e)}")
            return False
    
    tasks = [add_batch(batch, idx + 1) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #Count Successful batchs
    successful = sum(1 for result in results if result is True)
    if successful == len(batches):
        log_success(f"VectorStore Indexing: Completed indexing. Successfully indexed {successful}/{len(batches)} batches.")
    else:
        log_warning(f"VectorStore Indexing: Completed with some errors. Successfully indexed {successful}/{len(batches)} batches.")


async def main():
    """Main function to run the ingestion process."""
    log_header("DOCUMENTATION HELPER - INGESTION PIPELINE")

    # Scraping the website and extracting content
    log_info(" TavilCrawl: Starting web crawling documentation from https://python.langchain.com", Colors.PURPLE)

    #crawl the website and get the list of URLs
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 5,
        "extract_depth": "advanced"
    })
    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]
    log_success(f" TavilCrawl: Completed crawling. Total documents extracted: {len(all_docs)}")

    # Splitting the documents into smaller chunks
    log_header("DOCUMMENT CHUNKING")
    log_info(" RecursiveCharacterTextSplitter: Starting document chunking", Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents.")

    # Processing the documents and adding to vector store

    await index_documents_async(splitted_docs, batch_size=10)

    log_header("INGESTION PIPELINE COMPLETED")
    log_success("All documents have been processed and indexed successfully!")
    log_info("Summary:", Colors.BOLD)
    log_info(f" Total Documents Extracted: {len(all_docs)}", Colors.GREEN)
    log_info(f" Total Chunks Created: {len(splitted_docs)}", Colors.GREEN)

if __name__ == "__main__":
    asyncio.run(main())
