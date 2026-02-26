# wiki_html_markdown.py

import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from typing import Optional

from utils.logger import WiseryLogger

EXCLUDED_H2_HEADERS = {
    "See also", "Notes", "References", "External links", "Further reading", "Bibliography"
}


def get_html(title: str, language: str = "en") -> str:
    """
    Fetch the raw HTML content of a Wikipedia page using the REST API.

    Args:
        title (str): Title of the Wikipedia page.
        language (str, optional): Wikipedia language domain. Defaults to 'en'.

    Returns:
        str: HTML content of the Wikipedia page.

    Raises:
        requests.HTTPError: If the page could not be fetched.
    """
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/html/{title.replace(' ', '_')}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def clean_markdown(text: str) -> str:
    """
    Normalize excess newlines in markdown output.

    Args:
        text (str): Raw markdown text.

    Returns:
        str: Cleaned markdown text.
    """
    return re.sub(r'\n{3,}', '\n\n', text.strip())


def convert_html_to_ordered_markdown(html: str) -> str:
    """
    Convert Wikipedia HTML to markdown with inline tables.

    Args:
        html (str): Raw HTML from Wikipedia.

    Returns:
        str: Markdown-formatted content.
    """
    soup = BeautifulSoup(html, 'html.parser')
    output = []

    stop_parsing = False
    for tag in soup.body.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'table'], recursive=True):
        if tag.name in {'h2', 'h3'}:
            header_text = tag.get_text(strip=True).lower()
            if any(ex.lower() == header_text for ex in EXCLUDED_H2_HEADERS):
                stop_parsing = True
        if stop_parsing:
            break

        if tag.name == 'table':
            try:
                df = pd.read_html(str(tag))[0]
                md_table = df.to_markdown(index=False)
                output.append(md_table)
            except Exception as e:
                output.append(f"[TABLE PARSE ERROR] {e}")
        else:
            markdown = md(str(tag), heading_style="ATX")
            output.append(clean_markdown(markdown))

    return '\n\n'.join(output)


def get_wiki_page_with_md_tables_no_appendices(title: str, language: str = 'en', LOGGER: Optional[WiseryLogger] = None) -> str:
    """
    Retrieve a Wikipedia page as markdown, excluding standard appendix sections.

    Args:
        title (str): Wikipedia page title.
        language (str, optional): Language edition of Wikipedia. Defaults to 'en'.
        LOGGER (Optional[WiseryLogger], optional): Logger for info messages.

    Returns:
        str: Markdown-formatted Wikipedia content.
    """
    markdown = ""
    try:
        html = get_html(title, language=language)
        markdown = convert_html_to_ordered_markdown(html)
        if LOGGER:
            LOGGER.info(f"The page '{title}' exists in {language} Wikipedia.")
    except Exception as e:
        if LOGGER:
            LOGGER.info(f"The page '{title}' does not exist in {language} Wikipedia: {e}")
    return markdown


def save_wiki_page_to_txt(title: str, save_dir: str, LOGGER: Optional[WiseryLogger] = None) -> None:
    """
    Save a Wikipedia page (as markdown with tables) to a .txt file.

    Args:
        title (str): Wikipedia page title.
        save_dir (str): Directory path where the file should be saved.
        LOGGER (Optional[WiseryLogger], optional): Logger to use. Defaults to None.
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        html = get_html(title)
        markdown = convert_html_to_ordered_markdown(html)
        safe_title = title.replace('/', '_')
        filepath = os.path.join(save_dir, f"{safe_title}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)
        if LOGGER:
            LOGGER.info(f"Saved '{title}' to {filepath}")
        else:
            print(f"Saved to {filepath}")
    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Failed to save '{title}': {e}")
        else:
            print(f"Failed to save {title}: {e}")
