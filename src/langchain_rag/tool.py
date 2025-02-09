import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.documents import Document
from markdownify import markdownify as md
from retrying import retry
from tqdm import tqdm


@retry(wait_fixed=5000)
def ddgs_urls(query: str, max_results: int = 10) -> list[str]:
    """
    Retrieve a list of URLs from DuckDuckGo search results for a given query,
    filtered to exclude URLs containing 'api'.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of search results to retrieve.

    Returns:
        list[str]: A list of URLs.
    """
    full_query = f"{query} site:python.langchain.com"
    results = DDGS().text(full_query, max_results=max_results)
    # Use a set comprehension to remove duplicate URLs and filter out unwanted ones
    urls = list({result["href"] for result in results if "api" not in result["href"]})
    return urls


def page_content(url: str, session: requests.Session) -> Document | None:
    """
    Fetch and process the content of a web page.

    This function retrieves the page content from the given URL using the provided session,
    extracts the main article content, removes image tags, converts the content to markdown,
    and wraps it in a Document object.

    Args:
        url (str): The URL of the web page to fetch.
        session (requests.Session): The session object to use for HTTP requests.

    Returns:
        Document | None: A Document containing the page content and metadata if successful,
                         None otherwise.
    """
    try:
        response = session.get(url, timeout=10)
    except Exception as e:
        logging.error("Exception while fetching URL %s: %s", url, e)
        return None

    if response.status_code != 200:
        logging.warning(
            "Failed to fetch URL %s: Status code %s", url, response.status_code
        )
        return None

    article = BeautifulSoup(response.text, "html.parser").find("article")
    if not article:
        logging.warning("Article tag not found in URL: %s", url)
        return None

    img_tags = article.find_all("img")  # type: ignore
    logging.debug("Decomposing %d img tags in URL: %s", len(img_tags), url)
    for tag in img_tags:
        tag.decompose()

    content_md = md(str(article))
    if len(content_md) > 10_00_000:
        logging.warning("Article is too lengthy")
        return None

    return Document(page_content=content_md, metadata={"url": url})


def get_page_contents(urls: list[str], session: requests.Session) -> list[Document]:
    """
    Concurrently fetch and process the content of multiple web pages.

    Args:
        urls (list[str]): A list of URLs to fetch.
        session (requests.Session): The session object to use for HTTP requests.

    Returns:
        list[Document]: A list of Document objects containing the page content and metadata.
    """
    docs: list[Document] = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Map each future to its corresponding URL for better error logging
        future_to_url = {
            executor.submit(page_content, url, session): url for url in urls
        }
        for future in tqdm(
            as_completed(future_to_url), desc="Scraping URLs", total=len(future_to_url)
        ):
            try:
                result = future.result()
                if result is not None:
                    docs.append(result)
            except Exception as e:
                failed_url = future_to_url[future]
                logging.error("Error processing URL %s: %s", failed_url, e)
    return docs


def search_tool(query: str) -> list[Document]:
    """
    Search for and retrieve documents related to a query from python.langchain.com.

    This function performs the following steps:
      1. Queries DuckDuckGo (via DDGS) to retrieve URLs relevant to the given query,
         filtered to exclude API endpoints.
      2. Concurrently fetches the content of the retrieved URLs.
      3. Extracts the main article content, removes image tags, converts the content to markdown,
         and encapsulates it in Document objects.

    Args:
        query (str): The search query string.

    Returns:
        list[Document]: A list of Document objects containing the page content and metadata.
    """
    logging.info("Starting search for query: %s", query)
    urls = ddgs_urls(query)
    logging.info("Retrieved %d URLs", len(urls))
    with requests.Session() as session:
        docs = get_page_contents(urls, session)
    logging.info("Finished processing pages. Retrieved %d documents", len(docs))
    return docs


if __name__ == "__main__":
    user_query = "what is langchain?"
    documents = search_tool(user_query)
    # Log the URLs of the retrieved documents at the INFO level and detailed content at DEBUG level
    for doc in documents:
        logging.info("Document URL: %s", doc.metadata.get("url"))
        logging.debug("Document content: %s", doc.page_content)
