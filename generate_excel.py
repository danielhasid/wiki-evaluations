from correctness.final_answer_correctness import (
    compute_final_answer_correctness_metrics,
    export_to_excel_final_answer_correctness_info,
)

records = compute_final_answer_correctness_metrics(
    conversation_db="",
    db_filter={"step_id": None},
)

print(f"Scored {len(records)} records:")
for r in records:
    scores_str = ", ".join(f"{k}={v:.3f}" for k, v in r['scores'].items())
    print(f"  [{scores_str}]  query={r['query'][:70]!r}")

export_to_excel_final_answer_correctness_info("wiki_eval", records)
print("Excel written: wiki_eval_final_answer_correctness.xlsx")
