import os
import json
import random
import asyncio

from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

from adapters.llm_adapter import LLMInvoker, ModelConfig
from wikipedia.json_utils import parse_json_string
import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = config.LLM_MODEL or "gpt-4o"
GENERATED_TOKEN_LIMIT = 16384
CONTEXT_WINDOW = 128000
model = ChatOpenAI(
    model_name=model_name,
    temperature=0.9,
    n=1,
    frequency_penalty=0.05,
    api_key=config.OPENAI_API_KEY,
)
random.seed(42)

prompt_template = """
You are an intelligent assistant working with Wikipedia data.

Your task is to generate **deep and complex questions** based on the following Wikipedia pages content:

{contents}

For each question, provide the following JSON structure:

{{
  "question": "A complex, insightful question that can be answered using Wikipedia pages from the above categories.",
  "type": "summarization / aggregation / comparative / fact / other",
  "answer": "A full and well-explained answer derived from the content of the listed Wikipedia pages.",
  "supported_sentences": [
    {{
      "answer_sentence": "A sentence from the answer you provided.",
      "wiki_sentence": "An **explicit full sentence/sentences copied from Wikipedia pages content** that supports the answer.",
      "wiki_files": "files containing the answer"
    }}
    // You may add more supported_sentences as needed
  ]
}}

Rules:
- The `type` field should indicate the kind of question:
  * "summarization": Questions that explicitly request a condensed overview of the content. These should start with or include phrases such as "summarize," "describe in points," "outline the main concepts," or "give me a brief overview of."
  * "aggregation": Questions that involve numerical summaries or compiling lists from the text.
  * "comparative": Questions that compare and contrast elements within the text or across multiple texts.
  * "fact": Questions about specific facts from the text, such as names, dates, numbers, or explicit statements.
  * "other": Feel free to add additional question types if you discover interesting patterns or reasoning types not covered above. Clearly label and explain the new type if used.

- The `wiki_sentence` must be an **exact full sentence copied verbatim** from the Wikipedia page (verbatim).
    - Do not shorten or modify the sentence.
    - Always preserve full punctuation and structure as it appears in the source.
- Questions must be **unique**, not generic or repetitive.
- The number of questions you generate should depend on the **quantity and quality of the categories** provided.
  - If many broad or rich categories are provided, return more questions (up to 20).
  - If only a few or very narrow categories are given, limit to fewer questions (minimum 1).
- Choose content that requires synthesis across multiple files when possible.
- Prioritize questions that require thoughtful analysis, address significant consequences, and include multiple layers of complexity.
"""


def generate_questions_from_wiki_categories(model, prompt: str):
    """
    Generates questions based on a given text using a pre-trained model.

    Args:
        model: LangChain chat model.
        prompt (str): The formatted prompt text.

    Returns:
        list: A list of parsed question dicts.
    """
    try:
        prompt_msg = [HumanMessage(content=prompt)]
        responses = model.generate([prompt_msg])
        questions = [parse_json_string(response.text) for response in responses.generations[0]]
    except Exception as e:
        print(f"Failed to generate question: {e}")
        questions = []
    return questions


async def process_prompts(prompts, prompts_tokens_len):
    async_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.9,
        n=1,
        frequency_penalty=0.05,
        api_key=config.OPENAI_API_KEY,
    )

    tasks = []
    for prompt, num_tokens in zip(prompts, prompts_tokens_len):
        async_model.max_tokens = min(GENERATED_TOKEN_LIMIT, CONTEXT_WINDOW - num_tokens)
        tasks.append(async_model.agenerate([[HumanMessage(content=prompt)]]))

    return await asyncio.gather(*tasks)


def file_loader(file_path: str):
    """
    Simple file loader that reads a text file and returns its content
    as a plain string (replaces AppServer's file_loader).
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


if __name__ == '__main__':
    path = '/Users/yahel.salomon/Downloads/Darknet market'
    files_text = []
    file_names = []

    for file_name in os.listdir(path):
        print(file_name)
        try:
            text = file_loader(os.path.join(path, file_name))
            files_text.append(text)
            file_names.append(file_name)
        except Exception:
            pass

    contents_text = [
        'File name: ' + file_name + '\n' + '\n'.join(['\t' + row for row in text.split('\n')])
        for file_name, text in zip(file_names, files_text)
    ]
    random.shuffle(contents_text)

    content_in_batches = LLMInvoker.get_prompts_batches(
        model_id=ModelConfig.PROVIDER_MODEL_MISTRAL_LARGE_2407,
        prompts=contents_text,
        max_output_tokens=8192,
        context_window=128000 - 500,
    )

    contents = []
    for content_batch in content_in_batches:
        contents.append('\n\n'.join([f"{str(i + 1)}. {content}" for i, content in enumerate(content_batch)]))

    prompts = []
    prompts_tokens_len = []
    for i, (content, batch) in enumerate(zip(contents, content_in_batches)):
        formatted_prompt = prompt_template.format(contents=content)
        number_of_tokens = LLMInvoker.get_number_of_tokens(
            model_id=ModelConfig.PROVIDER_MODEL_MISTRAL_LARGE_2407,
            prompts=formatted_prompt,
        )
        if number_of_tokens > CONTEXT_WINDOW:
            print(f"Warning content number {i} exceeds context window")
        else:
            print(f"- Number of contents: {len(batch)}")
            print(f"- Number of tokens:   {number_of_tokens}")
            prompts.append(formatted_prompt)
            prompts_tokens_len.append(number_of_tokens)

    results = asyncio.run(process_prompts(prompts, prompts_tokens_len))
    results = [parse_json_string(response.generations[0][0].text) for response in results]
    questions_str = json.dumps(results, ensure_ascii=False, indent=2)
    with open(path + '.json', 'w') as f:
        f.write(questions_str)

    print(results)
