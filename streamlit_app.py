import os
import re
import io
import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="How Well does ChatGPT Cite Your Content?", layout="centered")

st.title("How Well does ChatGPT Cite Your Content?")
st.write("Upload a CSV with a **Prompt** column. I’ll call `gpt-4o-search-preview` per row, extract headline/publisher/date/URL as JSON, and give you a downloadable CSV.")

# ------------- OpenAI client -------------
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in environment. Add it to your .env or environment variables.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------- Helpers -------------
def extract_first_json_blob(text: str) -> Dict[str, Any]:
    """Find and parse the first JSON object in the text. Returns {} if parsing fails."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    blob = match.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        cleaned = blob.strip().strip("`")
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

def get_annotations_list(message) -> List[Any]:
    """Return a list of annotations from the SDK object, robust to attribute/dict access."""
    anns = getattr(message, "annotations", None)
    if anns is None and isinstance(message, dict):
        anns = message.get("annotations", None)
    return anns or []

def collect_url_citations(annotations) -> List[str]:
    """Collect URL citations as 'Title: URL' strings."""
    cites = []
    for ann in annotations:
        ann_type = getattr(ann, "type", None) if hasattr(ann, "type") else ann.get("type")
        url_cit = getattr(ann, "url_citation", None) if hasattr(ann, "url_citation") else ann.get("url_citation")
        if ann_type == "url_citation" and url_cit:
            title = getattr(url_cit, "title", None) if hasattr(url_cit, "title") else url_cit.get("title", "")
            url = getattr(url_cit, "url", None) if hasattr(url_cit, "url") else url_cit.get("url", "")
            if url:
                cites.append(f"{title}: {url}" if title else url)
    return cites

def call_model_for_prompt(prompt_text: str,
                          user_location: Dict[str, Any] | None,
                          search_context_size: str | None) -> Dict[str, Any]:
    """Calls gpt-4o-search-preview; returns response text, citations, and extracted fields."""
    system_instruction = (
        "You are given a quote or snippet. Identify the corresponding article and "
        "return ONLY a strict JSON object with these exact keys:\n"
        '{\n'
        '  "headline": string,\n'
        '  "publisher": string,\n'
        '  "publication_date": string,\n'
        '  "url": string\n'
        '}\n'
        "Do not include any commentary, explanations, markdown, or extra fields—just the JSON."
    )

    user_instruction = f"""
'{prompt_text}'
Identify the corresponding article’s headline, original publisher, publication date, and URL.
Return ONLY a JSON object with keys: headline, publisher, publication_date, url.
"""

    web_search_options: Dict[str, Any] = {}
    if user_location:
        web_search_options["user_location"] = {"type": "approximate", "approximate": user_location}
    if search_context_size in {"low", "medium", "high"}:
        web_search_options["search_context_size"] = search_context_size

    completion = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options=web_search_options,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction},
        ],
        # If supported in your environment, you can enforce JSON:
        # response_format={"type": "json_object"},
    )

    message = completion.choices[0].message
    response_text = message.content

    annotations = get_annotations_list(message)
    citations = collect_url_citations(annotations)

    parsed = extract_first_json_blob(response_text)
    extracted = {
        "headline": parsed.get("headline", ""),
        "publisher": parsed.get("publisher", ""),
        "publication_date": parsed.get("publication_date", ""),
        "url": parsed.get("url", ""),
    }

    return {"response_text": response_text, "citations": citations, "extracted": extracted}

# ------------- Sidebar options -------------
st.sidebar.header("Search Options")
col_country, col_city = st.sidebar.columns(2)
country = col_country.text_input("Country (ISO-2)", value="")
city = col_city.text_input("City", value="")
region = st.sidebar.text_input("Region/State", value="")
tz_hint = st.sidebar.text_input("Timezone (IANA, optional)", value="")
context_size = st.sidebar.selectbox("Search context size", ["medium", "low", "high"], index=0)

user_location = None
if country or city or region:
    user_location = {"country": country or None, "city": city or None, "region": region or None}
if tz_hint:
    # not formally used by the API right now, but keeping for your reference
    user_location = user_location or {}
    user_location["timezone"] = tz_hint

rate_limit = st.sidebar.number_input("Delay between calls (seconds)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

# ------------- File upload -------------
uploaded = st.file_uploader("Upload CSV with a 'Prompt' column", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    if "Prompt" not in df.columns:
        st.error("CSV must contain a 'Prompt' column.")
        st.stop()

    st.success(f"Loaded {len(df)} rows.")
    with st.expander("Preview input", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    if st.button("Run extraction", type="primary"):
        results: List[Dict[str, Any]] = []
        progress = st.progress(0, text="Processing…")

        for idx, row in df.iterrows():
            prompt = row["Prompt"]

            try:
                outcome = call_model_for_prompt(
                    prompt_text=prompt,
                    user_location=user_location,
                    search_context_size=context_size,
                )
            except Exception as e:
                # Capture errors and move on
                outcome = {
                    "response_text": f"ERROR: {e}",
                    "citations": [],
                    "extracted": {"headline": "", "publisher": "", "publication_date": "", "url": ""},
                }

            results.append({
                # Original columns (preserved; fill missing keys safely)
                "Publication": row.get("Publication", ""),
                "Source URL": row.get("Source URL", ""),
                "Prompt": row["Prompt"],
                "Date of Article": row.get("Date of Article", ""),
                # Raw model output & citations
                "Response": outcome["response_text"],
                "Citations": "; ".join(outcome["citations"]),
                # Extracted fields
                "Extracted Headline": outcome["extracted"]["headline"],
                "Extracted Publisher": outcome["extracted"]["publisher"],
                "Extracted Publication Date": outcome["extracted"]["publication_date"],
                "Extracted URL": outcome["extracted"]["url"],
            })

            # Rate limiting
            if rate_limit and rate_limit > 0:
                import time
                time.sleep(rate_limit)

            progress.progress((idx + 1) / len(df), text=f"Processed {idx + 1} / {len(df)}")

        output_df = pd.DataFrame(results)

        st.subheader("Result (first 50 rows shown)")
        st.dataframe(output_df.head(50), use_container_width=True)

        # Save to buffer and offer download
        csv_buf = io.StringIO()
        output_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name="responses_with_citations_and_fields.csv",
            mime="text/csv",
        )

        st.info("Done! Your CSV is ready to download.")

else:
    st.info("Upload a CSV to begin.")

