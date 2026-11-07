"""Strict classification tasks with explicit task-local label lists."""
import json

TASK_INSTRUCTIONS = {
 'constructcie': 'Classify the primary accident mechanism described in this construction accident narrative.',
 'osha': 'Predict the OSHA injury nature category from the narrative alone. Labels are administrative codes and may not be fully inferable from the text.'
}


def prompt(row, classes):
    allowed=classes[row['task']]
    instruction=(f"{TASK_INSTRUCTIONS[row['task']]}\nAllowed labels: {json.dumps(allowed)}\n"
                 'Return only one JSON object with exactly one field named "label". '
                 'The value must be one allowed label; do not include explanations.\n'
                 f"Narrative:\n{row['narrative']}")
    return instruction


def chat_prefix(tokenizer, row, classes):
    return tokenizer.apply_chat_template([{'role':'user','content':prompt(row,classes)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def gold(row):
    return json.dumps({'label':row['label']},ensure_ascii=False)
