import os
import json

path = '/Users/yahel.salomon/Downloads/QA_FILES_2'
write = 0

if write:
    file_names = os.listdir(path)

    file_paths = [os.path.join(path, file_name) for file_name in file_names]

    qa_data = []
    for file_name, file_path in zip(file_names, file_paths):
        if file_path.endswith('.json'):
            with open(file_path, "r") as f:
                page_qa_data = json.load(f)
                page_qa_data = [{'wiki_page': os.path.splitext(file_name)[0]} | page_qa for page_qa in page_qa_data]
                qa_data.extend(page_qa_data)
            f.close()

    with open(path + '.json', 'w', encoding='utf-8-sig') as f:
        json_str = json.dumps(qa_data, ensure_ascii=False, indent=2)
        f.write(json_str)
    f.close()
else:
    with open(path + '.json', "r", encoding='utf-8-sig') as f:
        qa_data = json.load(f)
    f.close()

for sample in qa_data:
    for sub_answer in sample['supported_sentences']:
        wiki_file_path = os.path.join('/Users/yahel.salomon/Downloads', 'Wikipedia_QA_per_Page_and_References',
                                      sample['wiki_page'], sub_answer['wiki_files'])
        with open(wiki_file_path, "r", encoding='utf-8-sig') as f:
            context = f.read()
        f.close()
