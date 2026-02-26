import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field, Extra
from utils.logger import WiseryLogger
import adapters.db_adapter as db_controller
from adapters.llm_adapter import execute_prompt, PromptType
from typing import List, Dict, TypedDict, Optional, Any


class ExportRecord(TypedDict):
    query: str
    response: str
    ground_truth: str
    score: float


class NvAccuracyScoreWO(BaseModel):
    rating_score: float = Field(
        description="A numeric rating score, must be one of: 4, 2, or 0.")

    class Config:
        extra = Extra.forbid


LOGGER = WiseryLogger().get_logger()

GOLDENSET_FRAMES = 'correctness_frames'
FULL_FRAMES_COLLECTION = "full_frames_with_entities"
EVAL_COLLECTION = 'final_answer_correctness'
EXCEL_CORRECTNESS = EVAL_COLLECTION + ".xlsx"


def load_ground_truth_frames(
    golden_set_db: str = GOLDENSET_FRAMES,
    collection_name: str = FULL_FRAMES_COLLECTION,
) -> pd.DataFrame:
    """
    Load the FRAMES ground truth dataset from MongoDB.

    Args:
        golden_set_db (str): The database name containing the ground truth collection.
        collection_name (str): actual data collection name

    Returns:
        pd.DataFrame: DataFrame containing the ground truth data.

    Raises:
        Exception: If no data is found.
    """
    LOGGER.info("Loading ground truth frames...")
    ground_truth_db = db_controller.get_db_data(db_name=golden_set_db, collection_name=collection_name)
    if not ground_truth_db:
        raise Exception(f"Ground truth not found in DB '{golden_set_db}', collection '{collection_name}'.")
    return pd.DataFrame.from_dict(ground_truth_db)


def load_user_queries(conversation_db: str, db_filter: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load user queries from the 'messages' collection.

    Args:
        conversation_db (str): Database name containing user messages.
        db_filter (Optional[Dict[str, Any]]): Mongo filter for selecting specific messages.

    Returns:
        pd.DataFrame: DataFrame of user messages.

    Raises:
        Exception: If no messages are found.
    """
    LOGGER.info("Loading user queries...")
    user_messages_db = db_controller.get_db_data(
        db_name=conversation_db,
        collection_name='messages',
        filter_term=db_filter or {}
    )
    if not user_messages_db:
        raise Exception(f"No user queries found in DB '{conversation_db}', collection 'messages'.")
    return pd.DataFrame.from_dict(user_messages_db)


def compute_final_answer_correctness_metrics(
        conversation_db: str,
        db_filter: Optional[Dict[str, Any]] = None,
        **kwargs
) -> List[ExportRecord]:
    """
    Compute correctness metrics for user responses compared to ground truth answers.

    Args:
        conversation_db (str): Database with user responses.
        db_filter (Optional[Dict[str, Any]]): Optional Mongo filter for messages.

    Returns:
        List[ExportRecord]: List of evaluated records with scores.
    """
    LOGGER.info("Starting correctness metrics computation...")

    ground_truth = load_ground_truth_frames()
    filtered_messages = load_user_queries(conversation_db, db_filter)

    export_records: List[ExportRecord] = []
    for _, msg_doc in filtered_messages.iterrows():
        record = evaluate_query_ragas(msg_doc, ground_truth, NvAccuracyScoreWO)
        if record:
            export_records.append(record)

    LOGGER.info("Completed correctness metrics computation.")
    return export_records


def evaluate_query_ragas(
        msg_doc: pd.Series,
        ground_truth: pd.DataFrame,
        pydantic_object: Any
) -> Optional[ExportRecord]:
    """
    Evaluate a single query and response against the ground truth.

    Args:
        msg_doc (pd.Series): User message document.
        ground_truth (pd.DataFrame): Ground truth DataFrame.
        pydantic_object (Any): Pydantic validation model.

    Returns:
        Optional[ExportRecord]: Evaluation record or None if no match.
    """
    query = msg_doc["query"]
    response = msg_doc["response"]
    matching_rows = ground_truth[ground_truth["Prompt"] == query]

    if matching_rows.empty:
        LOGGER.warning(f"No ground truth found for query: {query}")
        return None

    ground_truth_str = str(matching_rows.iloc[0]["Answer"])

    if response == "I could not find an answer to this query":
        rating_score = 0.0
    else:
        score1 = examine_final_answer_prompt(query, ground_truth_str, response, pydantic_object)
        score2 = examine_final_answer_prompt(query, response, ground_truth_str, pydantic_object)
        rating_score = (score1 + score2) / 8.0

    LOGGER.info(f"Query: {query} | Score: {rating_score}")
    return ExportRecord(
        query=query,
        response=response,
        ground_truth=ground_truth_str,
        score=rating_score
    )


def examine_final_answer_prompt(
        query: str,
        expected_answer: str,
        actual_answer: str,
        pydantic_object: Any
) -> float:
    """
    Execute the LLM prompt for answer comparison and parse its score.

    Args:
        query (str): The query.
        expected_answer (str): Ground truth answer.
        actual_answer (str): Model's answer.
        pydantic_object (Any): Pydantic validation model.

    Returns:
        float: The numeric score.
    """
    LOGGER.info(f"Comparing answers: expected='{expected_answer}' | actual='{actual_answer}'")
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    output_format = parser.get_format_instructions()

    result = execute_prompt(
        prompt_type=PromptType.RAGAS_WO_NV_ACCURACY,
        query=query,
        sentence_true=expected_answer,
        sentence_inference=actual_answer,
        output_format=output_format
    )
    nv_accuracy = parser.invoke(result)
    LOGGER.debug(f"LLM parsed score: {nv_accuracy}")
    return float(nv_accuracy['rating_score'])


def export_results(records: List[ExportRecord], conversation_db: str) -> None:
    """
    Export the evaluation results to MongoDB and an Excel file.

    Args:
        records (List[ExportRecord]): Evaluated records to export.
        conversation_db (str): Conversation DB name.
    """
    LOGGER.info("Exporting results to DB and Excel...")
    try:
        db_controller.insert_docs(conversation_db, EVAL_COLLECTION, records)
    except Exception as e:
        LOGGER.exception(f"DB export failed: {e}")

    try:
        export_to_excel_final_answer_correctness_info(conversation_db, records)
    except Exception as e:
        LOGGER.exception(f"Excel export failed: {e}")

    LOGGER.info("Export completed.")


def export_to_excel_final_answer_correctness_info(conversation_db: str, records: List[ExportRecord]) -> None:
    """
    Export evaluation results to an Excel file.

    Args:
        conversation_db (str): Conversation DB name (used in filename).
        records (List[ExportRecord]): Records to save.
    """
    LOGGER.info("Creating Excel export...")
    df = pd.json_normalize(records)
    avg_row = {
        'score': df['score'].mean(),
        'query': 'Average',
        'response': '',
        'ground_truth': '',
        '_id': ''
    }

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df.to_excel(conversation_db + "_" + EXCEL_CORRECTNESS, index=False)
    LOGGER.info(f"Excel file saved: {EXCEL_CORRECTNESS}")


def test_compute_final_answer_correctness_metrics(conversation_db_name: str) -> None:
    """
    Test run for computing correctness metrics and exporting them.
    """
    records = compute_final_answer_correctness_metrics(
        conversation_db=conversation_db_name,
        db_filter={"step_id": None}
    )
    export_results(records=records, conversation_db=conversation_db_name)


if __name__ == '__main__':
    test_compute_final_answer_correctness_metrics("p51dx4qju7d78miksie3j3tmpe")
