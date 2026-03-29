import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI

# -------------------------------
# FastAPI app setup
# -------------------------------
app = FastAPI(title="Image Uploader + OCR")
OCR_API_URL = "https://api.ocr.space/parse/image"
OCR_API_KEY = ""  # FIX 1: no hardcoded key
DEFAULT_MODEL = "gpt-4o-mini"
SESSION_STATE_PATH = Path("uploads/session_state.json")

DEFAULT_OPENAI_API_KEY = ""

# Allow browser calls from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Helper functions
# -------------------------------
def get_ocr_api_key() -> str:
    """Fetch the OCR API key from env — no hardcoded fallback."""
    key = OCR_API_KEY
    if not key:
        raise HTTPException(status_code=500, detail="OCR_API_KEY is not configured.")
    return key


def get_openai_client() -> OpenAI:
    """Build OpenAI client from env variable only — no hardcoded key."""
    api_key = DEFAULT_OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=api_key)


async def perform_ocr(file: UploadFile) -> str:
    """Forward uploaded image to OCR.space API and return parsed text."""
    await file.seek(0)  # FIX 6: reset stream in case it was partially read
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    payload = {
        "apikey": get_ocr_api_key(),  # FIX 1: no hardcoded key
        "language": "eng",
        "isOverlayRequired": False,
    }

    try:
        response = requests.post(
            OCR_API_URL,
            data=payload,
            files={
                "file": (
                    file.filename,
                    file_bytes,
                    file.content_type or "application/octet-stream",
                )
            },
            timeout=30,
        )
        response.raise_for_status()
        ocr_data = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OCR service error: {exc}") from exc

    if ocr_data.get("IsErroredOnProcessing"):
        message = ocr_data.get("ErrorMessage") or "OCR processing failed."
        raise HTTPException(status_code=400, detail=message)

    parsed_results = ocr_data.get("ParsedResults") or []
    text_blocks = [result.get("ParsedText", "") for result in parsed_results]
    parsed_text = "\n".join(text_blocks).strip()

    if not parsed_text:
        parsed_text = "(No text detected.)"

    return parsed_text


def run_expiry_prompt(ocr_text: str, model_name: Optional[str] = None) -> Dict[str, str]:
    """Send OCR text to OpenAI and get a JSON mapping of food items to expiry dates."""
    client = get_openai_client()
    today = date.today().isoformat()
    model_to_use = model_name or DEFAULT_MODEL

    messages = [
        {
            "role": "system",
            "content": (
                "You are a food-safety assistant. Infer likely expiry dates from grocery "
                "labels. If a date is written, use it. Otherwise, estimate using typical "
                "home life assuming the purchase date is today. Pick the primary grocery "
                "keyword when multiple words appear (e.g., 'tomato grape' → tomato; 'string "
                "cheese' → cheese). Double-check each expiry date for realism using common "
                "home-life ranges assuming proper storage (e.g., bread ~7 days, fresh milk "
                "~10–28 days refrigerated, fresh chicken ~1–2 weeks in the fridge, eggs "
                "~3–5 weeks refrigerated, hard cheese ~3–6 weeks, shelf-stable condiments "
                "like ketchup or mustard typically 2–4 months after opening if no date is "
                "present). Adjust estimates if they look too early/late. "
                "Never return N/A or null—always provide an ISO date (YYYY-MM-DD) estimate. "
                "Make sure to take into account conditments like ketchup, mustard, and penaut butter that are shelf-stable and can last much longer than fresh items. "
                "Ignore items that are not everyday grocery/food "
                "items. Also extract the receipt's total amount: pick the numeric value that "
                "appears multiple times near the bottom of typical receipts (tax + total) and "
                "label it clearly. Respond with raw JSON only — no markdown, no code fences. "
                "Return the following structure: {\"items\": {<item>: <expiry_date>, ...}, "
                "\"receipt_total\": "
                "<number_or_string> }."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Today is {today}. Parse the OCR text into a JSON object where keys are "
                "food item names and values are ISO dates (YYYY-MM-DD) for expected expiry. "
                "Also return the receipt total (number that repeats in the OCR, often the "
                "grand total). Output shape: {\"items\": {item: date}, \"receipt_total\": "
                "<number_or_string>}. Skip non-grocery items. OCR text:\n\n"
                f"{ocr_text}"
            ),
        },
    ]

    try:
        # FIX 2: use the correct SDK method
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
        )

        # FIX 3: correct response shape for chat completions
        text = response.choices[0].message.content.strip()

        # FIX 4: strip markdown code fences if the model wraps output in them
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        result = json.loads(text)

        # FIX 5: print the parsed dictionary clearly to the terminal
        # Fixed
        items_only = result.get("items", {})
        for item, expiry in items_only.items():
            print(f"  {item}: {expiry}")
        print()

        return result

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenAI returned unparseable JSON: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenAI request failed: {exc}"
        ) from exc


def run_meal_ideas_prompt(
    items_with_expiry: Dict[str, str], model_name: Optional[str] = None
) -> Dict[str, object]:
    """Generate 3 home-cookable meal ideas with inline recipes using receipt items."""
    client = get_openai_client()
    model_to_use = model_name or DEFAULT_MODEL

    # Provide the LLM a compact, deterministic snapshot of the items/expiries
    items_payload = json.dumps(items_with_expiry, ensure_ascii=False)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful home-cooking assistant. Create exactly 3 realistic, "
                "home-cookable meal ideas that primarily use the provided groceries "
                "(prioritize items expiring soon). Keep prep simple for a home kitchen. "
                "Return raw JSON only, no markdown, matching this shape: "
                '{"meals": [{"meal": "<name>", "uses": ["item1", "item2"], '
                '"recipe": {"summary": "<1-2 line overview>", '
                '"steps": ["step 1", "step 2", "step 3"]}}]}. '
                "Each recipe must be short (5-8 concise steps), pantry-friendly, and tailored to the given items. "
                "Do not include external links or markdown. Do not omit the recipe or steps."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Today is {date.today().isoformat()}. Here is a dictionary of items with "
                f"their expiry dates: {items_payload}. Suggest 3 meal ideas as specified."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
        )

        text = response.choices[0].message.content.strip()

        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        result = json.loads(text)
        meals = result.get("meals")
        if not isinstance(meals, list) or len(meals) == 0:
            raise ValueError("No meals returned from model")

        return result

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenAI returned unparseable meal JSON: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenAI meal idea request failed: {exc}"
        ) from exc


# -------------------------------
# Routes
# -------------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)) -> Dict[str, str]:
    """Accept uploaded image and save to ./uploads/"""
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    target_path = uploads_dir / file.filename
    with target_path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    return {
        "filename": file.filename,
        "content_type": file.content_type or "unknown",
        "saved_to": str(target_path),
    }


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)) -> JSONResponse:
    """Return extracted text from uploaded image."""
    parsed_text = await perform_ocr(file)
    return JSONResponse({"filename": file.filename, "text": parsed_text})


@app.post("/ocr/expiry")
async def ocr_to_expiry(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None, description="OpenAI model name"),  # FIX 7
) -> JSONResponse:
    """Extract OCR text and get estimated expiry dates using OpenAI."""
    parsed_text = await perform_ocr(file)

    if parsed_text == "(No text detected.)":
        print(f"[ocr/expiry] No text detected in {file.filename}")
        return JSONResponse(
            {
                "filename": file.filename,
                "raw_text": parsed_text,
                "estimated_items": {},
                "model": model or DEFAULT_MODEL,
            }
        )

    estimated = run_expiry_prompt(parsed_text, model_name=model)

    return JSONResponse(
        {
            "filename": file.filename,
            "raw_text": parsed_text,
            "estimated_items": estimated,
            "model": model or DEFAULT_MODEL,
        }
    )


@app.get("/healthz")
async def healthcheck() -> JSONResponse:
    return JSONResponse({"status": "ok"})

@app.post("/session/store")
async def store_session(data: Dict):
    global session_data
    session_data["items"] = data.get("items", [])
    session_data["total"] = data.get("total")
    session_data["cumulativeTotal"] = data.get("cumulativeTotal", 0)
    return {"status": "success"}

@app.get("/session/state")
async def get_session():
    return session_data

@app.post("/session/store")
async def store_session_state(payload: dict) -> JSONResponse:
    """Persist the latest expiry/items snapshot so dashboard/expiry can reload from backend."""
    SESSION_STATE_PATH.parent.mkdir(exist_ok=True, parents=True)
    try:
        with SESSION_STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist session state: {exc}") from exc
    return JSONResponse({"stored": True})


@app.get("/session/state")
async def get_session_state() -> JSONResponse:
    """Return persisted expiry/items snapshot if present."""
    if not SESSION_STATE_PATH.exists():
        return JSONResponse({"items": [], "total": None, "cumulativeTotal": None})
    try:
        with SESSION_STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read session state: {exc}") from exc
    return JSONResponse(data)


@app.post("/meals/from-expiry")
async def meal_ideas_from_expiry(payload: dict) -> JSONResponse:
    """Call ChatGPT to suggest 3 meal ideas based on the latest receipt items."""

    def extract_items(source) -> Dict[str, str]:
        extracted: Dict[str, str] = {}
        if isinstance(source, list):
            for entry in source:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or entry.get("item")
                expiry = entry.get("expiry") or entry.get("date") or ""
                if name:
                    extracted[name] = expiry
        elif isinstance(source, dict):
            extracted = {k: v or "" for k, v in source.items() if k}
        return extracted

    items_dict = extract_items(payload.get("items"))

    if not items_dict and SESSION_STATE_PATH.exists():
        try:
            with SESSION_STATE_PATH.open("r", encoding="utf-8") as f:
                stored = json.load(f)
            items_dict = extract_items(stored.get("items"))
        except (OSError, json.JSONDecodeError):
            items_dict = {}

    if not items_dict:
        raise HTTPException(status_code=400, detail="No items provided to build meal ideas.")

    ideas = run_meal_ideas_prompt(items_dict, model_name=payload.get("model"))
    return JSONResponse(ideas)


# -------------------------------
# Run server:
# uvicorn main:app --reload
# -------------------------------
