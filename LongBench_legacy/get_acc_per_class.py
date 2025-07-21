import os
import json
from pathlib import Path
from statistics import mean

category_map = {
    "narrativeqa": "QA",
    "qasper": "QA",
    "multifieldqa_en": "QA",
    "multifieldqa_zh": "QA",
    "hotpotqa": "QA",
    "2wikimqa": "QA",
    "musique": "QA",
    "dureader": "QA",
    "gov_report": "Summarization",
    "qmsum": "Summarization",
    "multi_news": "Summarization",
    "vcsum": "Summarization",
    "trec": "Few-shot Learning",
    "triviaqa": "Few-shot Learning",
    "samsum": "Few-shot Learning",
    "lsht": "Few-shot Learning",
    "passage_count": "Synthetic Task",
    "passage_retrieval_en": "Synthetic Task",
    "passage_retrieval_zh": "Synthetic Task",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion"
}

root_path = Path("pred/meta-llama-main")

for subfolder in root_path.iterdir():
    if subfolder.is_dir():
        result_file = subfolder / "result.json"
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            type_scores = {}
            for dataset, score in data.items():
                dataset_key = dataset.lower()
                if dataset_key in category_map:
                    task_type = category_map[dataset_key]
                    type_scores.setdefault(task_type, []).append(score)

            averaged_scores = {task: round(mean(scores), 2) for task, scores in type_scores.items()}

            overall_average = round(mean([score for scores in type_scores.values() for score in scores]), 2)
            averaged_scores["Overall Average"] = overall_average

            output_file = subfolder / "average_per_type.json"
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(averaged_scores, f_out, indent=4)

