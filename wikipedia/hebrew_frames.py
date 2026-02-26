import ast
import urllib.parse
import multiprocessing
import re

import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import wikipediaapi

import adapters.db_adapter as db_controller
from utils.logger import WiseryLogger
from wikipedia.wiki_html_markdown import get_wiki_page_with_md_tables_no_appendices

LOGGER = WiseryLogger().get_logger()

# ---------------------------------------------------------------------------
# Ground-truth FRAMES helpers (originally from final_answer_correctness.py)
# ---------------------------------------------------------------------------

GOLDENSET_FRAMES = 'correctness_frames'
FULL_FRAMES_COLLECTION = "full_frames_with_entities"


def load_ground_truth_frames(
    golden_set_db: str = GOLDENSET_FRAMES,
    collection_name: str = FULL_FRAMES_COLLECTION,
) -> pd.DataFrame:
    """Load the FRAMES ground truth dataset from MongoDB."""
    LOGGER.info("Loading ground truth frames...")
    ground_truth_db = db_controller.get_db_data(db_name=golden_set_db, collection_name=collection_name)
    if not ground_truth_db:
        raise Exception(f"Ground truth not found in DB '{golden_set_db}', collection '{collection_name}'.")
    return pd.DataFrame.from_dict(ground_truth_db)


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

def extract_title(url):
    """
    Extracts and decodes the Wikipedia title from a given URL.

    Parameters:
        url (str): The Wikipedia URL.

    Returns:
        str: Cleaned article title with underscores replaced by spaces.
    """
    path = url.split("/wiki/")[-1]
    return path.split("#")[0].replace("_", " ")


def sanitize_filename(title):
    """
    Replaces invalid filename characters with underscores to make the title filesystem-safe.

    Parameters:
        title (str): Wikipedia page title.

    Returns:
        str: Sanitized title safe for filenames.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", title)


def translate_text(text, src_lang='auto', dest_lang='en'):
    """
    Translates text from source language to destination language using Google Translate.

    Parameters:
    - text (str): The text to translate.
    - src_lang (str): Source language code (e.g., 'fr', 'es', 'auto' to auto-detect).
    - dest_lang (str): Destination language code (e.g., 'en', 'de', 'ja').

    Returns:
    - str: Translated text.
    """
    LOGGER.info("start")
    try:
        translator = Translator()
        translation = translator.translate(text, src=src_lang, dest=dest_lang)
        return translation.text
    except Exception as e:
        LOGGER.error(f"Translation failed: {e}")

    return ''


def get_wikipedia_page_content(url, language='en'):
    """
    Given a Wikipedia page URL, returns the plain text content of the page.

    Parameters:
    - url (str): Full URL of the Wikipedia page.
    - language (str): Language code for the Wikipedia (default is 'en').

    Returns:
    - page: Wikipedia page.
    """
    LOGGER.info("start")
    page = None
    try:
        if "/wiki/" not in url:
            return "Invalid Wikipedia URL"
        title_part = url.split("/wiki/")[1].split('#')[0]
        title = urllib.parse.unquote(title_part).replace('_', ' ')

        wiki = wikipediaapi.Wikipedia(user_agent='CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)',
                                      language=language)
        page = wiki.page(title)

        if page.exists():
            LOGGER.info(f"The page '{title}' exist in {language} Wikipedia.")
        else:
            LOGGER.info(f"The page '{title}' does not exist in {language} Wikipedia.")

        return page
    except Exception as e:
        LOGGER.error(f"Failed to fetch content: {e}")

    LOGGER.info("end")
    return page


def translate_long_text(text, src_lang='auto', dest_lang='en', max_chunk_size=1024 * 4):
    """
    Translates a long text from source language to destination language by chunking it.

    Parameters:
    - text (str): The long text to translate.
    - src_lang (str): Source language code (e.g., 'auto', 'fr').
    - dest_lang (str): Destination language code (e.g., 'en', 'de').
    - max_chunk_size (int): Max characters per translation chunk (Google limit is ~5000).

    Returns:
    - str: The translated text.
    """
    LOGGER.info("start")
    try:
        translator = Translator()
        chunks = []
        current_chunk = ""

        for paragraph in text.split('\n'):
            if len(current_chunk) + len(paragraph) + 1 < max_chunk_size:
                current_chunk += paragraph + '\n'
            else:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n'
        if current_chunk:
            chunks.append(current_chunk.strip())

        translated_chunks = []
        for chunk in chunks:
            translated = translator.translate(chunk, src=src_lang, dest=dest_lang)
            translated_chunks.append(translated.text)

        return '\n\n'.join(translated_chunks)

    except Exception as e:
        LOGGER.error(f"Translation failed: {e}")

    return ''


def translate_wiki_link_md(link, dest_lang) -> dict:
    link_pages_text_translated = {}
    title = extract_title(link)
    text_link_page = get_wiki_page_with_md_tables_no_appendices(title=title, language=dest_lang, LOGGER=LOGGER)
    safe_title = sanitize_filename(title)

    if not text_link_page:
        text_link_page = get_wiki_page_with_md_tables_no_appendices(title=title, LOGGER=LOGGER)
        link_page_text_translated = translate_long_text(text_link_page, dest_lang=dest_lang)
        link_page_title_translated = translate_long_text(safe_title, dest_lang=dest_lang)
        link_pages_text_translated[link] = {
            'text': link_page_text_translated,
            'title': link_page_title_translated
        }
    else:
        link_pages_text_translated[link] = {
            'text': text_link_page,
            'title': safe_title
        }

    return link_pages_text_translated


def translate_wiki_link(link, dest_lang) -> dict:
    link_pages_text_translated = {}

    link_page = get_wikipedia_page_content(url=link, language=dest_lang)

    if not link_page or not link_page.text:
        link_page = get_wikipedia_page_content(url=link)
        link_page_text_translated = translate_long_text(link_page.text, dest_lang=dest_lang)
        link_page_title_translated = translate_long_text(link_page.title, dest_lang=dest_lang)
        link_pages_text_translated[link] = {
            'text': link_page_text_translated,
            'title': link_page_title_translated
        }
    else:
        link_pages_text_translated[link] = {
            'text': link_page.text,
            'title': link_page.title
        }

    return link_pages_text_translated


def translate_wiki_item(item: dict, dest_lang: str) -> dict:
    links = [(link, dest_lang) for link in ast.literal_eval(item['wiki_links'])]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(translate_wiki_link_md, links)

    return {key: value for dict_item in results for key, value in dict_item.items()}


def translate_frames(channel_id: str, dest_lang: str = 'he'):
    """
    Translates the FRAMES dataset Wikipedia pages to the target language.

    Input:
        Assumes the FRAMES dataset was uploaded to 'channel_id', under 'correctness_frames'.
        https://wiserylabs.atlassian.net/wiki/spaces/Research/pages/285704258/FRAMES+EN+HEB+Agent+and+Data

    Output:
        The translated frames will be created under f'correctness_frames_translated_{dest_lang}'
    """
    LOGGER.info("start")

    goldenset = 'correctness_frames'
    translated_goldenset = f'{goldenset}_translated_{dest_lang}'

    LOGGER.info(f"reading frames data from {channel_id}")
    ground_truth = load_ground_truth_frames()
    translated = pd.DataFrame()
    try:
        translated = load_ground_truth_frames(collection_name=translated_goldenset)
    except Exception as e:
        LOGGER.info(f"Failed to read {translated_goldenset}: {e}")

    frames_list = ground_truth.to_dict(orient='records')
    translated_goldenset_list = translated.to_dict(orient='records')
    translated_goldenset_dict = {}

    rephrased_frames = []

    for item in tqdm(frames_list):
        if item['Index'] not in translated_goldenset_dict.keys():
            item.pop('_id')

            LOGGER.info("rephrase prompt")
            query = item['Prompt']

            trans_query = translate_text(query, dest_lang=dest_lang)
            item['Prompt'] = trans_query

            answer = item['Answer']
            trans_answer = translate_text(answer, dest_lang=dest_lang)
            item['Answer'] = trans_answer

            link_pages_text_translated = translate_wiki_item(item, dest_lang)

            if link_pages_text_translated:
                item['translated_links'] = link_pages_text_translated

                LOGGER.info(f"writing query:\n\n{trans_query}\n\n to {translated_goldenset} data, in {channel_id}")
                db_controller.insert_docs(db_name=channel_id,
                                          collection_name=translated_goldenset,
                                          docs=[item])

            rephrased_frames.append(item)

    LOGGER.info(f"done creating {len(rephrased_frames)} translated queries")

    LOGGER.info("end")
    return rephrased_frames


if __name__ == "__main__":
    translate_frames(channel_id='3t3z7ftmt7do8mk6aebg3emexw')
