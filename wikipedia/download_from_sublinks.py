import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin


def download_pdf(pdf_url, output_dir):
    """
    Download a PDF file and save it to the specified directory.
    """
    if not pdf_url.startswith('http'):
        return False

    try:
        filename = pdf_url.split('/')[-1]
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        filename = re.sub(r'[\\/*?:"<>|]', "", filename)

        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"File already exists: {filename}")
            return True

        print(f"Downloading: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=30)

        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {filename}")
            return True
        else:
            print(f"Failed to download {pdf_url}: Not a valid PDF or server error")
            return False

    except Exception as e:
        print(f"Error downloading {pdf_url}: {str(e)}")
        return False


def get_publication_links(url):
    """
    Extracts links to individual publication pages from a listing page.
    """
    try:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')

        publications = []

        article_items = soup.find_all('article') or soup.find_all('div', class_='views-row')

        for item in article_items:
            links = item.find_all('a')
            for link in links:
                href = link.get('href')
                if href and not href.startswith(('http://', 'https://', 'mailto:')):
                    full_url = urljoin(url, href)
                    if full_url not in publications:
                        publications.append(full_url)

        return publications
    except Exception as e:
        print(f"Error extracting publication links from {url}: {str(e)}")
        return []


def extract_pdf_links(publication_url):
    """
    Extracts PDF download links from a publication page.
    """
    try:
        response = requests.get(publication_url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')

        pdf_links = []

        for link in soup.find_all('a'):
            href = link.get('href')
            if not href:
                continue

            is_pdf = (
                    href.lower().endswith('.pdf') or
                    'pdf' in link.get_text().lower() or
                    'download' in link.get_text().lower() or
                    'download' in href.lower() or
                    'attachment' in href.lower()
            )

            if is_pdf:
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(publication_url, href)

                if href not in pdf_links:
                    pdf_links.append(href)

        return pdf_links
    except Exception as e:
        print(f"Error extracting PDF links from {publication_url}: {str(e)}")
        return []


def get_next_page_url(current_url):
    """
    Find the URL for the next page of publications.
    """
    try:
        response = requests.get(current_url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')

        pager = soup.find('ul', class_='pager')
        if pager:
            next_link = pager.find('a', string='next') or pager.find('a', string='Next')
            if next_link:
                href = next_link.get('href')
                if href:
                    return urljoin(current_url, href)

        if 'page=' in current_url:
            current_page = int(re.search(r'page=(\d+)', current_url).group(1))
            next_page = current_page + 1
            return re.sub(r'page=\d+', f'page={next_page}', current_url)

        return None
    except Exception as e:
        print(f"Error getting next page URL from {current_url}: {str(e)}")
        return None


def main():
    base_url = "https://www.understandingwar.org/publications?page=1"
    output_dir = "/Users/yahel.salomon/Downloads/understandingwar/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_url = base_url
    page_count = 1
    total_pdfs_downloaded = 0

    while current_url:
        print(f"\nProcessing page {page_count}: {current_url}")

        publication_links = get_publication_links(current_url)
        print(f"Found {len(publication_links)} publication links")

        page_pdfs_downloaded = 0

        for pub_link in publication_links:
            print(f"Checking publication: {pub_link}")

            pdf_links = extract_pdf_links(pub_link)
            print(f"Found {len(pdf_links)} PDF links")

            for pdf_link in pdf_links:
                success = download_pdf(pdf_link, output_dir)
                if success:
                    page_pdfs_downloaded += 1
                    total_pdfs_downloaded += 1

                time.sleep(1)

            time.sleep(2)

        print(f"Downloaded {page_pdfs_downloaded} PDFs from page {page_count}")

        next_url = get_next_page_url(current_url)
        if next_url and next_url != current_url:
            current_url = next_url
            page_count += 1
            time.sleep(3)
        else:
            break

    print(f"\nDownload complete. Total PDFs downloaded: {total_pdfs_downloaded}")


if __name__ == "__main__":
    main()
