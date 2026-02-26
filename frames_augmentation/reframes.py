import pandas as pd
from tqdm import tqdm

import adapters.db_adapter as db_controller
from utils.logger import WiseryLogger
from adapters.llm_adapter import execute_prompt, PromptType
from wikipedia.hebrew_frames import load_ground_truth_frames

LOGGER = WiseryLogger().get_logger()

if __name__ == "__main__":
    LOGGER.info("start")

    # Assuming FRAMES dataset was uploaded to this channel, under 'correctness_frames'
    # https://wiserylabs.atlassian.net/wiki/spaces/Research/pages/285704258/FRAMES+EN+HEB+Agent+and+Data
    channel_id = 'h477og5xnpnnbj8id7wkpffgfy'
    goldenset = 'correctness_frames'
    rephrased_goldenset = f'{goldenset}_rephrased'

    # Index, Prompt, Answer, wiki_links, reasoning_types, wikipedia_link_1,..., wiki_link
    LOGGER.info(f"reading frames data from {channel_id}")
    ground_truth = load_ground_truth_frames()
    existing_rephrased = pd.DataFrame()
    try:
        existing_rephrased = load_ground_truth_frames(collection_name=rephrased_goldenset)
    except Exception as e:
        LOGGER.info(f"Failed to read {rephrased_goldenset}: {e}")

    frames_list = ground_truth.to_dict(orient='records')
    rephrased_frames_list = existing_rephrased.to_dict(orient='records')
    rephrased_frames_dict = {item['Index']: item for item in rephrased_frames_list}

    rephrased_frames = []
    for item in tqdm(frames_list):
        if item['Index'] not in rephrased_frames_dict:
            item.pop('_id')

            LOGGER.info("rephrase prompt")
            query = item['Prompt']

            rephrased_prompt = execute_prompt(prompt_type=PromptType.FRAMES_REPHRASE, query=query)

            item['Prompt'] = rephrased_prompt

            LOGGER.info(f"writing:\n\n{rephrased_prompt}\n\n to {rephrased_goldenset} data to {channel_id}")
            db_controller.insert_docs(db_name=channel_id,
                                      collection_name=rephrased_goldenset,
                                      docs=[item])

            rephrased_frames.append(item)

    LOGGER.info(f"done creating {len(rephrased_frames)} rephrased queries")

    LOGGER.info("end")
