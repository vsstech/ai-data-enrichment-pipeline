import json
import math
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

from product_variant_fetcher_utils import fetch_product_variants

MODEL_CLASSIFIER = os.getenv("CLASSIFIER_MODEL", "gpt-5.4-mini")
MODEL_JUDGE = os.getenv("JUDGE_MODEL", "gpt-5.4")
INPUT_CSV = os.getenv("INPUT_CSV", "historical_products.csv")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "historical_products_taxonomy_validated.csv")
PRODUCT_NAME_COL = os.getenv("PRODUCT_NAME_COL", "historical_product_name")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "1.0"))
CONTEXT_DATA_CSV = "electronic_retailer_taxonomy_dataset.csv"

load_dotenv()
client = OpenAI()


def build_schema_and_examples() -> Dict[str, str]:
    context_data_csv = pd.read_csv(CONTEXT_DATA_CSV)

    brands = sorted(context_data_csv["brand"].dropna().unique().tolist())
    families = sorted(context_data_csv["product_family"].dropna().unique().tolist())
    categories = sorted(context_data_csv["taxonomy_category"].dropna().unique().tolist())
    types = sorted(context_data_csv["product_type"].dropna().unique().tolist())

    schema_text = f"""
brand: one of {brands}

product_family: one of {families}

taxonomy_category: one of {categories}

product_type: one of {types}

variants: object with optional keys:
  - storage or capacity: string or null
  - color: string or null
  - size: string or null
""".strip()

    examples = []
    sample = context_data_csv.sample(min(5, len(context_data_csv)), random_state=0)
    for _, row in sample.iterrows():
        name = str(row["product_name"])
        ex = (
            "Input: \"" + name + "\"\n"
            "Output:\n" +
            "{\n"
            f"  \"input_name\": \"{name}\",\n"
            f"  \"brand\": \"{row['brand']}\",\n"
            f"  \"product_family\": \"{row['product_family']}\",\n"
            f"  \"taxonomy_category\": \"{row['taxonomy_category']}\",\n"
            f"  \"product_type\": \"{row['product_type']}\",\n"
            "  \"variants\": {},\n"
            "  \"confidence\": 0.99,\n"
            "  \"reasoning\": \"Example from existing taxonomy.\"\n"
            "}\n"
        )
        examples.append(ex)

    return {"schema": schema_text, "examples": "\n".join(examples)}


SCHEMA_AND_EXAMPLES = build_schema_and_examples()

CLASSIFIER_SYSTEM_PROMPT = f"""
You are a strict product taxonomy classifier.

You receive historical or EOL product names and optional retrieved web evidence.
You must map each one into an EXISTING taxonomy used for Apple and Samsung products.

Rules:
1. Use ONLY the allowed values listed in the schema below.
2. Do NOT invent new product_families, taxonomy_categories, or product_types.
3. Use web evidence only as supporting context for variants and product identification.
4. If the product name itself lacks variants, you MAY use retrieved official web evidence.
5. If a variant value is not supported by product name or retrieved evidence, omit it.
6. Output MUST be valid JSON.
7. Put short reasoning inside a "reasoning" field.

<schema>
{SCHEMA_AND_EXAMPLES['schema']}
</schema>

<examples>
{SCHEMA_AND_EXAMPLES['examples']}
</examples>

Return a JSON array with fields:
- input_name
- brand
- product_family
- taxonomy_category
- product_type
- variants
- confidence
- reasoning
""".strip()

JUDGE_SYSTEM_PROMPT = """
You are a senior product taxonomy QA reviewer.

Evaluate the output of a classification model that maps historical product names
into an Apple/Samsung taxonomy and variants schema.

Rules:
1. Use ONLY the input_name, retrieved evidence, and prediction JSON.
2. Check that brand, product_family, taxonomy_category, and product_type are logically consistent.
3. Check that variants are supported by product name or retrieved evidence.
4. If any field is clearly wrong, mark invalid.
5. Be conservative.

Return a JSON array in the same order, each item with:
- valid: boolean
- score: number between 0 and 1
- errors: list of strings
- comments: short string
""".strip()


def chunked(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def normalize_variants(v: Any) -> str:
    if not isinstance(v, dict):
        return "{}"
    cleaned = {k: val for k, val in v.items() if val not in (None, "", "None")}
    return json.dumps(cleaned, ensure_ascii=False)


def brand_hint_from_name(product_name: str):
    p = product_name.lower()
    if p.startswith(("iphone", "ipad", "apple watch")):
        return "Apple"
    if p.startswith("galaxy"):
        return "Samsung"
    return None


def classify_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    user_prompt = "Classify the following historical products:\n\n" + json.dumps(items, ensure_ascii=False, indent=2)
    response = client.responses.create(
        model=MODEL_CLASSIFIER,
        input=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    data = json.loads(response.output_text.strip())
    if not isinstance(data, list):
        raise ValueError("Classifier response is not a JSON array")
    return data


def judge_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    user_prompt = "Judge the following classified products:\n\n" + json.dumps(items, ensure_ascii=False, indent=2)
    response = client.responses.create(
        model=MODEL_JUDGE,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    data = json.loads(response.output_text.strip())
    if not isinstance(data, list):
        raise ValueError("Judge response is not a JSON array")
    return data


def main():
    df = pd.read_csv(INPUT_CSV)
    if PRODUCT_NAME_COL not in df.columns:
        raise ValueError(f"Missing required column: {PRODUCT_NAME_COL}")

    enriched_items = []
    for name in df[PRODUCT_NAME_COL].fillna("").astype(str).str.strip().tolist():
        web_data = fetch_product_variants(product_name=name, brand_hint=brand_hint_from_name(name))
        enriched_items.append({
            "input_name": name,
            "retrieved_web_variants": web_data.get("variants", {}),
            "retrieved_web_source_url": web_data.get("source_url"),
            "retrieved_web_status": web_data.get("status"),
            "retrieved_web_evidence": web_data.get("evidence"),
        })

    predictions = []
    total_batches = math.ceil(len(enriched_items) / BATCH_SIZE) if enriched_items else 0
    for idx, batch in enumerate(chunked(enriched_items, BATCH_SIZE), start=1):
        print(f"Classifying batch {idx}/{total_batches} with {len(batch)} rows using {MODEL_CLASSIFIER}")
        predictions.extend(classify_batch(batch))
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    if len(predictions) != len(df):
        raise ValueError(f"Prediction row count mismatch. input={len(df)} predictions={len(predictions)}")

    judge_items = []
    for enriched, pred in zip(enriched_items, predictions):
        judge_items.append({
            "input_name": enriched["input_name"],
            "retrieved_web_variants": enriched["retrieved_web_variants"],
            "retrieved_web_source_url": enriched["retrieved_web_source_url"],
            "retrieved_web_status": enriched["retrieved_web_status"],
            "retrieved_web_evidence": enriched["retrieved_web_evidence"],
            "prediction": pred,
        })

    verdicts = []
    for idx, batch in enumerate(chunked(judge_items, BATCH_SIZE), start=1):
        print(f"Judging batch {idx}/{total_batches} with {len(batch)} rows using {MODEL_JUDGE}")
        verdicts.extend(judge_batch(batch))
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    if len(verdicts) != len(df):
        raise ValueError(f"Judge row count mismatch. input={len(df)} verdicts={len(verdicts)}")

    pred_df = pd.DataFrame(predictions)
    verdict_df = pd.DataFrame(verdicts)

    if "variants" not in pred_df.columns:
        pred_df["variants"] = "{}"
    pred_df["variants"] = pred_df["variants"].apply(normalize_variants)

    out = df.copy()
    out["retrieved_web_variants"] = [json.dumps(x.get("retrieved_web_variants", {}), ensure_ascii=False) for x in enriched_items]
    out["retrieved_web_source_url"] = [x.get("retrieved_web_source_url") for x in enriched_items]
    out["retrieved_web_status"] = [x.get("retrieved_web_status") for x in enriched_items]
    out["predicted_brand"] = pred_df.get("brand")
    out["predicted_product_family"] = pred_df.get("product_family")
    out["predicted_taxonomy_category"] = pred_df.get("taxonomy_category")
    out["predicted_product_type"] = pred_df.get("product_type")
    out["predicted_variants"] = pred_df.get("variants")
    out["prediction_confidence"] = pred_df.get("confidence")
    out["prediction_reasoning"] = pred_df.get("reasoning")
    out["judge_valid"] = verdict_df.get("valid")
    out["judge_score"] = verdict_df.get("score")
    out["judge_errors"] = verdict_df.get("errors").apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]")
    out["judge_comments"] = verdict_df.get("comments")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
