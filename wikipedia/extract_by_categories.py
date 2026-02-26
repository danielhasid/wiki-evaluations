import wikipediaapi
import numpy as np

# https://en.wikipedia.org/wiki/Category:Drugs


user_agent = 'CategoryExtractor/1.0 (https://github.com/your-username/wiki-extract)'
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent=user_agent
)


def get_wikipedia_page(title, file_path, lang='en', main_flag=False):
    wiki_wiki = wikipediaapi.Wikipedia(language=lang,
                                       user_agent='MyCoolScraper/1.0 (youremail@example.com)'
                                       )
    page = wiki_wiki.page(title)

    if not page.exists():
        print(f"Page '{title}' does not exist.")
        return None

    if main_flag:
        with open(file_path, 'w', encoding='utf-8') as f:
            full_text = str(page.fullurl + '\n\n' + page.title + '\n\n' + page.text)
            f.write(full_text)

    return {
        'url': page.fullurl,
        'title': page.title,
        'summary': page.summary,
        'full_text': page.text
    }


def extract_categories(category_name, n=0, N=2):
    # stop condition
    if n >= N:
        return []

    if category_name.startswith('Category:') or category_name.startswith('File:'):
        category = wiki.page(f'{category_name}')
    else:
        category = wiki.page(f'Category:{category_name}')

    categories = []
    for title, page in category.categorymembers.items():
        print(page.title)
        if 'File:' in str(page.title):
            categories.append(page.title)
        # Only include actual pages (not subcategories)
        if page.ns == wikipediaapi.Namespace.MAIN:
            categories.append(page.title)
        else:
            categories += extract_categories(page.title, n=n + 1)
    return categories


if __name__ == '__main__':
    topics = ["Bandidos_Motorcycle_Club", 'Drugs']
    file_path = '/Users/yahel.salomon/Downloads/Wikipedia_Illegal drug trade.txt'
    categories = extract_categories('Weapons')
    categories = list(np.unique(categories))
    print(categories)
