import json
from .schema import Extraction, MISSING_KEYS

SYSTEM = ("Extract only information explicitly stated in the following fictional intake narrative. "
          "Never infer an injury, treatment, location, date, or liability factor. "
          "Use null for absent date/location, empty lists for absent list values, "
          "and list absent field names in missing_information. "
          "Return ONLY a JSON object matching the requested schema, without markdown. "
          "Use concise phrases copied or normalized from the narrative.")
USER = ("Narrative:\n{narrative}\n\nJSON keys: incident_type (vehicle_collision, "
        "slip_and_fall, workplace_injury), incident_date (YYYY-MM-DD or null), "
        "location (string or null), injuries (string[]), medical_treatment (string[]), "
        "liability_factors (string[]), missing_information (field names from: "
        + ", ".join(MISSING_KEYS) + ").")

def messages(narrative, examples=()):
    out = [{"role": "system", "content": SYSTEM}]
    for ex in examples:
        out.append({"role": "user", "content": USER.format(narrative=ex["narrative"])})
        out.append({"role": "assistant", "content": json.dumps(ex["label"], sort_keys=True)})
    out.append({"role": "user", "content": USER.format(narrative=narrative)})
    return out

def template(tokenizer, narrative, examples=()):
    return tokenizer.apply_chat_template(messages(narrative, examples), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)

def target(label):
    return json.dumps(Extraction.model_validate(label).model_dump(mode="json"), sort_keys=True)
