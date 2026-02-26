from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import os
import string
import re
import random
import time

import requests

from wikipedia.extract_by_categories import get_wikipedia_page, extract_categories


def get_random_string(length: int) -> str:
    """
    Generates a random string of a specified length, consisting of ASCII letters and digits.

    Args:
        length (int): The desired length of the generated string.

    Returns:
        str: A randomly generated string of the specified length composed of ASCII letters (both uppercase and lowercase) and digits.
    """
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def get_wikipedia_references(page_title_or_url):
    """
    Get references from a Wikipedia page in simple format

    Args:
        page_title_or_url: Title or URL of the Wikipedia page

    Returns:
        list: List of references with their numbers and text
    """
    # Handle URL or title input
    if page_title_or_url.startswith('http'):
        url = page_title_or_url
    else:
        url = f"https://en.wikipedia.org/wiki/{page_title_or_url.replace(' ', '_')}"

    # Get page content
    response = requests.get(
        url,
        headers={'User-Agent': 'ReferenceExtractor/1.0'}
    )

    if response.status_code != 200:
        print(f"Error: Failed to fetch the page. Status code: {response.status_code}")
        return []

    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find references list - typically an ordered list with class "references"
    references_list = soup.find('ol', class_='references')

    if not references_list:
        print("Could not find references list on the page.")
        return []

    # Extract references
    references = []

    for i, li in enumerate(references_list.find_all('li'), 1):
        # Get reference text, removing any "jump up" links
        for span in li.select('.mw-jump-link'):
            span.decompose()

        # Extract text and URLs
        text = li.get_text(strip=True)
        urls = [a.get('href') for a in li.find_all('a', href=True) if a.get('href').startswith('http')]

        references.append({
            'number': i,
            'text': text,
            'urls': urls
        })

    return references


def download_pdf(url, response, output_folder, filename=None):
    """
    Simple function to download a PDF directly from a URL

    Args:
        url: URL of the PDF to download
        response: pre-fetched requests.Response object
        output_folder: Folder where to save the PDF
        filename: Optional filename to use (if None, extracts from URL)

    Returns:
        str: Path to the downloaded file if successful, None otherwise
    """
    os.makedirs(output_folder, exist_ok=True)

    try:
        if not filename:
            filename = url.split('/')[-1].split('?')[0]

            if not filename or '.' not in filename:
                filename = 'downloaded_document.pdf'

        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        file_path = os.path.join(output_folder, filename)

        if os.path.isfile(file_path):
            random_string = get_random_string(5)
            file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_' + random_string + '.pdf')

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"PDF successfully downloaded to: {file_path}")
        return file_path

    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None


def save_content(text: str, write_type: str, output_folder=None, filename=None):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
    else:
        filepath = filename
    with open(filepath, write_type, encoding='utf-8') as f:
        f.write(text)


def get_url_content(url, output_folder=None):
    """
    Get content from a URL with improved extraction including PDF support

    Args:
        url: URL to extract content from
        output_folder: Optional folder to save downloaded content

    Returns:
        dict: Title and text content from the URL
    """
    try:
        headers = {}
        print('before:', url)
        if url.endswith('view'):
            url = url.strip('view')
        elif url.endswith('download'):
            url = url.strip('download')
        if url.endswith('/'):
            url = url[:-1]

        print('after:', url)
        response = requests.get(url, headers=headers, timeout=100, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')

        if ('application/pdf' in content_type.lower()) or url.lower().endswith('.pdf') or url.lower().endswith(
                '/view') or url.lower().endswith('/download'):
            print('try download:', url)
            try:
                download_pdf(url, response, output_folder, filename=os.path.basename(url))
            except Exception:
                pass
            return {
                'url': url,
                'title': 'Archived PDF',
                'content': f"File {os.path.basename(url)} was downloaded",
                'content_type': content_type,
            }

        def extract_text():
            for tag in ['article', 'main', 'div']:
                section = soup.find(tag)
                if section:
                    ps = section.find_all('p')
                    if ps:
                        return '\n\n'.join(p.get_text(strip=True) for p in ps if p.get_text(strip=True))
            ps = soup.find_all('p')
            if ps:
                return '\n\n'.join(p.get_text(strip=True) for p in ps)
            return soup.get_text(strip=True)

        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else 'No title'

        article_div = soup.find('div', {'id': 'column1'})
        if not article_div:
            article_div = soup.find('div', class_='article-text')
        if article_div:
            paragraphs = article_div.find_all('p')
            content = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)
        else:
            content = url + '\n\n' + extract_text()

        filename = re.sub(r'[^a-zA-Z0-9]+', '_', title[:50]).strip('_') + '.txt'
        save_content(text=content, write_type='w', output_folder=output_folder, filename=filename)

        return {
            'url': url,
            'title': title,
            'content': content,
            'content_type': content_type,
        }

    except Exception as e:
        print(f"Request failed: {e}")
        return None


def get_clean_content(urls, output_folder=None):
    info = None
    for i, url in enumerate(urls):
        if info:
            break
        info = get_url_content(url, output_folder=output_folder)
        time.sleep(1)
    return info


def run_parallel(refs, output_folder=None):
    with Parallel(n_jobs=-1, verbose=0, backend='threading') as parallel:
        references_info = parallel(
            delayed(get_clean_content)(ref['urls'], output_folder=output_folder) for ref in refs if ref['urls']
        )
    return references_info


def run(refs, output_folder=None):
    references_info = []
    for ref in refs:
        if ref['urls']:
            output = get_clean_content(ref['urls'], output_folder=output_folder)
            references_info.append(output)
            time.sleep(1)
    return references_info


def extract_wiki_with_reference(entry_name, output_dir):
    output_folder = os.path.join(output_dir, entry_name)
    os.makedirs(output_folder, exist_ok=True)
    get_wikipedia_page(entry_name, file_path=os.path.join(str(output_folder), f'Wikipedia_{entry_name}.txt'),
                       main_flag=True)

    refs = get_wikipedia_references(entry_name)

    print(f"Found {len(refs)} references:")
    for ref in refs:
        print(f"{ref['number']}. {ref['text'][:100]}...")
        if ref['urls']:
            print(f"   Links: {ref['urls'][0]}")

    run(refs, output_folder=output_folder)


if __name__ == "__main__":
    output_dir = '/Users/yahel.salomon/Downloads'
    category_name = 'Weapons'
    categories = extract_categories(category_name)
    for category in categories:
        extract_wiki_with_reference(category, output_dir)
