from datetime import datetime
import ast
import random
import uuid
from typing import List
import pandas as pd

from utils.logger import WiseryLogger
import adapters.db_adapter as db_controller
from adapters.retrieval_adapter import (
    RetrievalType,
    RETRIEVAL_RANKING_FUNCTIONS,
    SystemConfig,
    EvaluationType,
    call_retrieval_function,
    store_to_mongo_aggregated_results,
    evaluate_single_db,
)
from correctness.final_answer_correctness import load_ground_truth_frames

LOGGER = WiseryLogger().get_logger()


def compute_correctness_metrics(channel_id: str, retrievers_names: List[str], agent_name: str, user_query: str,
                                k: int = 200,
                                **kwargs):
    """Receives a retrievers_names, query and agent and returns the correctness metrics for those arguments."""
    LOGGER.info("start")

    LOGGER.info("calling call_retrieval_function")
    docs_and_scores = call_retrieval_function(retrievers_names, agent_name, user_query, k)
    # Sort documents by score ascending (lower = more similar)
    sorted_docs = sorted(
        [(doc, score) for doc, score in docs_and_scores],
        key=lambda x: x[1]
    )

    retrieved_files = set()
    for rank, (doc, score) in enumerate(sorted_docs):
        retrieved_files.add(doc.metadata["file_name"])

    retrieved_files_list = list(retrieved_files)

    # GOLDEN SET is expected to be under collection_name='correctness_frames'
    # https://wiserylabs.atlassian.net/wiki/spaces/Research/pages/285704258/FRAMES+EN+HEB+Agent+and+Data
    ground_truth = load_ground_truth_frames()
    user_query_gt = ground_truth.loc[ground_truth['Prompt'] == user_query].to_dict('records')
    if user_query_gt:
        expected_pages_list = ast.literal_eval(user_query_gt[0]['wiki_links_entities'])
    else:
        expected_pages_list = []

    expected_pages = set(expected_pages_list)

    metrics = {
        'timestamp': datetime.now(),
        'user_query': user_query,
        'channel_id': channel_id,
        'retriever_name': str(retrievers_names),
        'agent_name': agent_name
    }
    LOGGER.info("calculate retrieve metrics")
    for k_item in [item for item in [10, 20, 30, 50, 100, 200] if item <= k]:
        try:
            LOGGER.info(f"len(expected_pages_list[:k_item]) {len(expected_pages_list[:k_item])}")
            retrieved_files_set_k = set(retrieved_files_list[:k_item])
            recall = 1.0
            if expected_pages_list[:k_item]:
                recall = (len(retrieved_files_set_k & expected_pages) * 1.0) / len(expected_pages_list[:k_item])
            precision = 0.0
            if retrieved_files_list[:k_item]:
                precision = (len(retrieved_files_set_k & expected_pages) * 1.0) / len(retrieved_files_list[:k_item])

            metrics[f'recall@{k_item}'] = recall
            metrics[f'precision@{k_item}'] = precision

            metrics[f'is_query_answered_pass@{k_item}'] = retrieved_files_set_k.issuperset(expected_pages)
        except Exception as e:
            LOGGER.info(f"Error processing k_item {k_item}: {e}")

    return_metrics = [metrics]

    LOGGER.info("end")
    return return_metrics


def enrich_sources_with_correctness_info(db_name: str = None, db_filter: dict = None, all_retrievers: List[str] = None,
                                         ranking_func: str = RETRIEVAL_RANKING_FUNCTIONS.DEFAULT.value, k: int = 200,
                                         **kwargs):
    """
    Receives a DB name (channel id), and enriches the sources with the correctness results.
    If DB name is None, run over all DBs in mongo, and search for the sources' collection.

    Should receive filter that will be used over the documents in the collection.

    If the key `evaluation` exists, then rename it to `old_evaluation`.

    # 1. filter docs from db.sources
    # 2.0 for each msg_id & task_id: extract agent_name, user query
    #   2.1 compute evaluation
    #   2.2. add it to all the docs from 2.0 (with update mongodb cmd)
    """
    LOGGER.info(f"start, k={k}")
    evaluation_id = str(uuid.uuid4())
    LOGGER.info(f"evaluation_id={evaluation_id}")
    valid_retrievers = [RetrievalType.RETRIEVE_RELEVANT_DOCS.value,
                        RetrievalType.VECTOR_DB_SEMANTIC.value,
                        RetrievalType.BM25_KEYWORDS.value,
                        RetrievalType.LLM_KEYWORDS.value,
                        RetrievalType.LOCATION_KEYWORDS.value]
    if not all_retrievers:
        all_retrievers = valid_retrievers
    if ranking_func:
        if ranking_func in [RETRIEVAL_RANKING_FUNCTIONS.RRF.value, RETRIEVAL_RANKING_FUNCTIONS.DEFAULT.value]:
            SystemConfig().set_config(config="ranking_function_name", new_value=RETRIEVAL_RANKING_FUNCTIONS.RRF.value)
        if ranking_func in [RETRIEVAL_RANKING_FUNCTIONS.RRF.value]:
            all_retrievers.append([str(item) for item in valid_retrievers])

    db_filter = db_filter if db_filter else {}
    if db_name:
        all_dbs = [db_name]
    else:
        num_of_dbs_to_test = int(SystemConfig().get_config("correctness_tests_db_limit"))
        all_dbs = [channel for channel in db_controller.get_all_db_names() if len(channel) == 26]
        all_dbs = random.sample(all_dbs, min([num_of_dbs_to_test, len(all_dbs)]))

    for db_name_i in all_dbs:
        LOGGER.info(f"running on channel: {db_name_i}")
        metric_dict = {'test_name': EvaluationType.CORRECTNESS.value,
                       'compute_metrics_func': compute_correctness_metrics}
        if not kwargs.get('calculate_only', False):
            evaluate_single_db(metric_dict=metric_dict, db_name=db_name_i, N=1, db_filter=db_filter,
                               all_retrievers=all_retrievers, evaluation_id=evaluation_id, k=k)

        LOGGER.info("correctness aggregated metrics calculation")
        all_results = store_to_mongo_aggregated_results(db_name=db_name,
                                                        all_retrievers=all_retrievers)
        if kwargs.get('write_to_excel', False):
            dfs = []
            all_results.pop('_id')
            for retriever_name in all_results.keys():
                dfs.append(pd.DataFrame.from_dict(
                    {key: item for key, item in all_results[retriever_name].items() if
                     key.find('@') > 0}))
                dfs[-1]['retriever'] = [retriever_name] * len(dfs[-1])
            pd.concat(dfs).to_excel(f"retrievers_{ranking_func}.xlsx")
    LOGGER.info("end")


if __name__ == '__main__':
    channel = 'h477og5xnpnnbj8id7wkpffgfy'
    channel = 'ys46fwnt1frkxrfnnajztz1t4w'
    goldenset = 'correctness_frames'

    enrich_sources_with_correctness_info(db_name='ys46fwnt1frkxrfnnajztz1t4x',
                                         ranking_func='reciprocal_rank_fusion',
                                         calculate_only=True,
                                         write_to_excel=True,
                                         )
