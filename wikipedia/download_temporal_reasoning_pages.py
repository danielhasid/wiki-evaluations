"""
Script to download Wikipedia pages from FRAMES dataset queries with 'Temporal reasoning'.

This script:
1. Loads the FRAMES dataset from Hugging Face
2. Filters queries containing 'Temporal reasoning' in reasoning_types
3. Extracts unique Wikipedia URLs from filtered queries
4. Downloads each Wikipedia page as plain text (.txt files)

Usage:
    python download_temporal_reasoning_pages.py
    
    # Optional: specify output directory
    python download_temporal_reasoning_pages.py --output-dir path/to/output
"""

import os
import urllib.parse
import re
import ast
import argparse
from typing import List, Set, Dict
from pathlib import Path

import pandas as pd
import wikipediaapi
from tqdm import tqdm

OUTPUT_DIR = Path("output/downloaded_wikipedia_pages_by_category")


def extract_title(url: str) -> str:
    """
    Extracts and decodes the Wikipedia title from a given URL.
    
    Parameters:
        url (str): The Wikipedia URL.
    
    Returns:
        str: Cleaned article title with underscores replaced by spaces.
    """
    path = url.split("/wiki/")[-1]
    title = path.split("#")[0].replace("_", " ")
    return urllib.parse.unquote(title)


def sanitize_filename(title: str) -> str:
    """
    Replaces invalid filename characters with underscores to make the title filesystem-safe.
    
    Parameters:
        title (str): Wikipedia page title.
    
    Returns:
        str: Sanitized title safe for filenames.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", title)


def load_frames_from_huggingface() -> List[Dict]:
    """
    Loads the FRAMES dataset from Hugging Face datasets repository.
    
    Returns:
        List[Dict]: List of dialogue entries from the dataset.
    """
    print("Loading FRAMES dataset from Hugging Face...")

    try:
        df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

        frames_data = df.to_dict(orient='records')

        print(f"Successfully loaded FRAMES dataset with {len(frames_data)} entries.")
        return frames_data

    except Exception as e:
        print(f"Error loading FRAMES dataset from Hugging Face: {e}")
        raise


def filter_temporal_reasoning_queries(frames_data: List[Dict], categories: List[str] = None) -> List[Dict]:
    """
    Filters queries that contain specified categories in their reasoning_types field.
    
    Parameters:
        frames_data (List[Dict]): The FRAMES dataset entries.
        categories (List[str]): List of reasoning categories to filter by. 
                                Defaults to ['Temporal reasoning'].
    
    Returns:
        List[Dict]: Filtered entries containing any of the specified categories.
    """
    if categories is None:
        categories = ['Temporal reasoning']

    categories_str = ", ".join(f"'{cat}'" for cat in categories)
    print(f"\nFiltering queries with categories: {categories_str}...")

    filtered_queries = []

    for entry in frames_data:
        reasoning_types = entry.get('reasoning_types', [])

        if isinstance(reasoning_types, str):
            try:
                reasoning_types = ast.literal_eval(reasoning_types)
            except (ValueError, SyntaxError):
                reasoning_types = [reasoning_types]

        if any(category in reasoning_types for category in categories):
            filtered_queries.append(entry)

    print(f"Found {len(filtered_queries)} queries matching the specified categories.")
    return filtered_queries


def extract_wikipedia_urls(filtered_queries: List[Dict]) -> Set[str]:
    """
    Extracts unique Wikipedia URLs from filtered queries.
    
    Parameters:
        filtered_queries (List[Dict]): Filtered query entries.
    
    Returns:
        Set[str]: Set of unique Wikipedia URLs.
    """
    print("\nExtracting Wikipedia URLs...")

    wikipedia_urls = set()

    for entry in filtered_queries:
        wiki_links = entry.get('wiki_links', [])

        if isinstance(wiki_links, str):
            try:
                wiki_links = ast.literal_eval(wiki_links)
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse wiki_links: {wiki_links}")
                continue

        if isinstance(wiki_links, list):
            for url in wiki_links:
                if url and 'wikipedia.org' in str(url):
                    wikipedia_urls.add(url)

        for key, value in entry.items():
            if key.startswith('wikipedia_link_') and value:
                if 'wikipedia.org' in str(value):
                    wikipedia_urls.add(value)

    print(f"Found {len(wikipedia_urls)} unique Wikipedia URLs.")
    return wikipedia_urls


def get_wikipedia_page_content(url: str, language: str = 'en') -> tuple:
    """
    Fetches Wikipedia page content using the wikipediaapi library.
    
    Parameters:
        url (str): Full URL of the Wikipedia page.
        language (str): Language code for Wikipedia (default is 'en').
    
    Returns:
        tuple: (title, text) - Wikipedia page title and content, or (None, None) if failed.
    """
    try:
        if "/wiki/" not in url:
            print(f"Invalid Wikipedia URL: {url}")
            return None, None

        title_part = url.split("/wiki/")[1].split('#')[0]
        title = urllib.parse.unquote(title_part).replace('_', ' ')

        wiki = wikipediaapi.Wikipedia(
            user_agent='FramesResearch/1.0 (https://example.org/frames; frames@example.org)',
            language=language
        )
        page = wiki.page(title)

        if page.exists():
            return page.title, page.text
        else:
            print(f"Page '{title}' does not exist in {language} Wikipedia.")
            return None, None

    except Exception as e:
        print(f"Error fetching content for {url}: {e}")
        return None, None


def download_wikipedia_pages(wikipedia_urls: Set[str], output_dir: Path):
    """
    Downloads Wikipedia pages as plain text files.
    
    Parameters:
        wikipedia_urls (Set[str]): Set of Wikipedia URLs to download.
        output_dir (Path): Directory to save the text files.
    """
    print(f"\nDownloading Wikipedia pages to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    successful_downloads = 0
    failed_downloads = 0

    for url in tqdm(list(wikipedia_urls), desc="Downloading pages"):
        title, text = get_wikipedia_page_content(url)

        if title and text:
            safe_filename = sanitize_filename(title) + ".txt"
            file_path = output_dir / safe_filename

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"{'-' * 80}\n\n")
                    f.write(text)

                successful_downloads += 1
            except Exception as e:
                print(f"Error saving file {safe_filename}: {e}")
                failed_downloads += 1
        else:
            failed_downloads += 1

    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {successful_downloads} pages")
    print(f"Failed downloads: {failed_downloads} pages")


def main(categories: List[str] = None):
    """
    Main function to orchestrate the download process.
    
    Parameters:
        categories (List[str]): List of reasoning categories to filter by.
                                Defaults to ['Temporal reasoning'].
    """
    if categories is None:
        categories = ['Temporal reasoning']

    parser = argparse.ArgumentParser(
        description='Download Wikipedia pages from FRAMES dataset with specified categories'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory for Wikipedia pages'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        default=None,
        help='Reasoning categories to filter by (default: Temporal reasoning)'
    )

    args = parser.parse_args()

    if args.categories is not None:
        categories = args.categories

    categories_str = ", ".join(f"'{cat}'" for cat in categories)
    print("=" * 80)
    print(f"FRAMES Dataset - Wikipedia Pages Downloader")
    print(f"Categories: {categories_str}")
    print("=" * 80)

    try:
        frames_data = load_frames_from_huggingface()

        filtered_queries = filter_temporal_reasoning_queries(frames_data, categories)

        if not filtered_queries:
            print(f"\nNo queries with categories {categories_str} found in the dataset.")
            return

        wikipedia_urls = extract_wikipedia_urls(filtered_queries)

        if not wikipedia_urls:
            print("\nNo Wikipedia URLs found in filtered queries.")
            return

        output_path = Path(args.output_dir)
        download_wikipedia_pages(wikipedia_urls, output_path)

        print("\n" + "=" * 80)
        print("Process completed successfully!")
        print(f"Wikipedia pages saved to: {output_path.absolute()}")
        print("=" * 80)

    except Exception as e:
        print(f"\nError in main process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
