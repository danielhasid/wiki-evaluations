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
    scores: Dict[str, float]


class NvAccuracyScoreWO(BaseModel):
    rating_score: float = Field(
        description="A numeric rating score, must be one of: 4, 2, or 0.")

    class Config:
        extra = Extra.forbid


LOGGER = WiseryLogger().get_logger()

EVAL_COLLECTION = 'final_answer_correctness'
EXCEL_CORRECTNESS = EVAL_COLLECTION + ".xlsx"

GROUND_TRUTH_CSV = "data/ground_truth_frames.csv"
MESSAGES_CSV = "data/messages.csv"


def load_ground_truth_frames(
    csv_path: str = GROUND_TRUTH_CSV,
) -> pd.DataFrame:
    """
    Load the FRAMES ground truth dataset from a CSV file.

    Args:
        csv_path (str): Path to the ground truth CSV file.
            Expected columns: Prompt, Answer, wiki_links_entities

    Returns:
        pd.DataFrame: DataFrame containing the ground truth data.

    Raises:
        Exception: If the file is missing or empty.
    """
    LOGGER.info(f"Loading ground truth frames from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise Exception(f"Ground truth CSV not found at '{csv_path}'.")
    if df.empty:
        raise Exception(f"Ground truth CSV at '{csv_path}' is empty.")
    return df


def load_user_queries(conversation_db: str = "", db_filter: Optional[Dict[str, Any]] = None,
                      csv_path: str = MESSAGES_CSV) -> pd.DataFrame:
    """
    Load user queries from a CSV file.

    Args:
        conversation_db (str): Unused — kept for backwards compatibility.
        db_filter (Optional[Dict[str, Any]]): Key/value pairs applied as equality
            filters on the loaded DataFrame columns (e.g. {"step_id": None}).
        csv_path (str): Path to the messages CSV file.
            Expected columns: query, response, step_id

    Returns:
        pd.DataFrame: DataFrame of user messages.

    Raises:
        Exception: If the file is missing or no rows remain after filtering.
    """
    LOGGER.info(f"Loading user queries from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise Exception(f"Messages CSV not found at '{csv_path}'.")

    if db_filter:
        for col, val in db_filter.items():
            if col in df.columns:
                if val is None:
                    df = df[df[col].isna()]
                else:
                    df = df[df[col] == val]

    if df.empty:
        raise Exception(f"No user queries found in '{csv_path}' after applying filter {db_filter}.")
    return df


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
        scores = {"ragas_accuracy": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0}
    else:
        # ragas_accuracy: bidirectional NV accuracy (normalised to 0-1)
        score1 = examine_final_answer_prompt(query, ground_truth_str, response, pydantic_object)
        score2 = examine_final_answer_prompt(query, response, ground_truth_str, pydantic_object)
        ragas_accuracy = (score1 + score2) / 8.0

        # answer_relevancy: does the response address the question? (normalised to 0-1)
        ar_raw = examine_final_answer_prompt(
            query, "", response, pydantic_object,
            prompt_type=PromptType.ANSWER_RELEVANCY
        )
        answer_relevancy = ar_raw / 4.0

        # faithfulness: is the response consistent with the ground truth? (normalised to 0-1)
        faith_raw = examine_final_answer_prompt(
            query, ground_truth_str, response, pydantic_object,
            prompt_type=PromptType.FAITHFULNESS
        )
        faithfulness = faith_raw / 4.0

        scores = {
            "ragas_accuracy": round(ragas_accuracy, 4),
            "answer_relevancy": round(answer_relevancy, 4),
            "faithfulness": round(faithfulness, 4),
        }

    LOGGER.info(f"Query: {query} | Scores: {scores}")
    return ExportRecord(
        query=query,
        response=response,
        ground_truth=ground_truth_str,
        scores=scores
    )


def examine_final_answer_prompt(
        query: str,
        expected_answer: str,
        actual_answer: str,
        pydantic_object: Any,
        prompt_type: PromptType = PromptType.RAGAS_WO_NV_ACCURACY,
) -> float:
    """
    Execute the LLM prompt for answer evaluation and parse its score.

    Args:
        query (str): The query.
        expected_answer (str): Ground truth answer (may be empty for relevancy-only prompts).
        actual_answer (str): Model's answer.
        pydantic_object (Any): Pydantic validation model.
        prompt_type (PromptType): Which prompt to use for evaluation.

    Returns:
        float: The raw numeric score returned by the LLM (0, 2, or 4).
    """
    LOGGER.info(f"[{prompt_type.value}] expected='{expected_answer}' | actual='{actual_answer}'")
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    output_format = parser.get_format_instructions()

    kwargs: Dict[str, Any] = dict(
        query=query,
        sentence_inference=actual_answer,
        output_format=output_format,
    )
    if prompt_type != PromptType.ANSWER_RELEVANCY:
        kwargs["sentence_true"] = expected_answer

    result = execute_prompt(prompt_type=prompt_type, **kwargs)
    parsed = parser.invoke(result)
    LOGGER.debug(f"LLM parsed score: {parsed}")
    return float(parsed['rating_score'])


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

    Columns: query | response | ground_truth | scores.<metric> ...
    A final 'Average' row shows the mean for every score column.

    Args:
        conversation_db (str): Conversation DB name (used in filename).
        records (List[ExportRecord]): Records to save.
    """
    LOGGER.info("Creating Excel export...")
    # pd.json_normalize flattens {"scores": {"ragas_accuracy": 1.0, ...}}
    # into columns named  scores.ragas_accuracy, scores.answer_relevancy, …
    df = pd.json_normalize(records)

    score_cols = [c for c in df.columns if c.startswith("scores.")]
    avg_row: Dict[str, Any] = {"query": "Average", "response": "", "ground_truth": ""}
    for col in score_cols:
        avg_row[col] = round(df[col].mean(), 4)

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Reorder: identity columns first, then score columns
    ordered_cols = ["query", "response", "ground_truth"] + score_cols
    df = df[[c for c in ordered_cols if c in df.columns]]

    filepath = conversation_db + "_" + EXCEL_CORRECTNESS
    df.to_excel(filepath, index=False)
    LOGGER.info(f"Excel file saved: {filepath}")


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
