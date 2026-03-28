import json
import os
from datetime import date
from pathlib import Path
from typing import Dict

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
DEFAULT_OCR_API_KEY = "K520b2d191988957"
DEFAULT_MODEL = "gpt-4o-mini"  # fallback model
DEFAULT_OPENAI_API_KEY = "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"

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
def get_api_key() -> str:
    """Fetch the OCR API key from env; fallback to default."""
    return os.getenv("OCR_API_KEY", DEFAULT_OCR_API_KEY)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=api_key)


async def perform_ocr(file: UploadFile) -> str:
    """Forward uploaded image to OCR.space API and return parsed text."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    payload = {
        "apikey": get_api_key(),
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


def run_expiry_prompt(ocr_text: str, model_name: str | None = None) -> Dict[str, str]:
    """Send OCR text to OpenAI and get a JSON mapping of food items to expiry dates, with debug prints."""
    client = get_openai_client()
    today = date.today().isoformat()
    model_to_use = model_name or DEFAULT_MODEL

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a food-safety assistant. Infer likely expiry dates from grocery "
                "labels. If a date is written, use it. Otherwise, estimate using typical "
                "shelf life assuming the purchase date is today."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Today is {today}. Parse the OCR text into a JSON object where keys are "
                "food item names and values are ISO dates (YYYY-MM-DD) for expected expiry. "
                "If unsure, return null for that item. OCR text:\n\n"
                f"{ocr_text}"
            ),
        },
    ]

    try:
        response = client.responses.create(
            model=model_to_use,
            input=prompt
        )

        # Get text output from the LLM
        text = getattr(response, "output_text", None)
        if not text:
            text = "".join(
                item["content"][0]["text"]
                for item in getattr(response, "output", [])
                if item["type"] == "message"
            )

        # Debug print: raw AI output
        print(f"[LLM raw output]: {text}")

        # Parse JSON
        estimated = json.loads(text)

        # Debug print: parsed dictionary
        print(f"[ocr/expiry] Parsed estimated items: {estimated}")

        return estimated

    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenAI request failed or returned unparseable JSON: {exc}"
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
    model: str | None = Form(default=None, description="OpenAI model name"),
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
    """Simple health check endpoint."""
    return JSONResponse({"status": "ok"})


# -------------------------------
# Run server:
# uvicorn main:app --reload
# -------------------------------