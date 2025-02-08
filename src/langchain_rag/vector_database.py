import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from markdownify import markdownify as md
from tqdm import tqdm

SITEMAP_URL = "https://python.langchain.com/sitemap.xml"


def request_response(url: str, session: requests.Session) -> str | None:
    try:
        response = session.get(url)
        if response.status_code != 200:
            response.raise_for_status
        return response.text
    except:
        return None


def get_urls(sitemap_url: str, session: requests.Session) -> list[str]:
    response = request_response(sitemap_url, session)
    if not response:
        raise ValueError(f"Unable to access URL: {sitemap_url}")
    xml_content = BeautifulSoup(response, "xml").find_all("loc")
    urls = [i.text for i in xml_content]
    return urls


def get_doc(url: str, session: requests.Session) -> Document | None:
    response = request_response(url, session)
    if not response:
        print(f"Unable to fetch page content: {url}")
        return None
    article_tag = BeautifulSoup(response, "html.parser").find("article")
    if not article_tag:
        print(f"Unable to find article tag: {url}")
        return None

    # Remove all image tags from the article_tag
    for img in article_tag.find_all("img"):  # type: ignore
        img.decompose()

    page_content = md(str(article_tag))
    doc = Document(page_content=page_content, metadata={"url": url})
    return doc


def get_docs(urls: list[str], session: requests.Session) -> list[Document]:
    docs = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_doc, url, session) for url in urls]
        for f in tqdm(
            as_completed(futures), desc="Scrapping URLs", total=len(urls), unit="urls"
        ):
            docs.append(f.result())
    docs = [i for i in docs if i]
    print(f"Total docs: {len(docs)}")
    return docs


def save_docs(docs: list[Document], path: str = "src/temp") -> None:
    full_path = os.path.join(path, "docs.pkl")
    with open(full_path, "wb") as f:
        pickle.dump(docs, f)
    print(f"Saved to path: {full_path}")


def main():
    with requests.Session() as session:
        urls = get_urls(SITEMAP_URL, session)
        docs = get_docs(urls, session)
    save_docs(docs)


if __name__ == "__main__":
    main()
