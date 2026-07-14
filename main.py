import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ocr import extract_text
from llm_service import analyze_medicine
from vector_store import init_vector_store

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Medicine Backside Checker")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.on_event("startup")
def startup():
    init_vector_store()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analyze")
def analyze_get():
    """Browsers land here via the address bar / refresh after a form POST;
    send them back to the upload form instead of a 405."""
    return RedirectResponse(url="/", status_code=303)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    manual_text: str = Form(default=""),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Unsupported file type. Upload JPEG, PNG, or WEBP.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    ocr_result = extract_text(image_bytes)
    manual_text = manual_text.strip()
    if manual_text:
        ocr_result["raw_text"] = f"{manual_text}\n{ocr_result['raw_text']}".strip()
        ocr_result["lines"] = [(manual_text, 1.0)] + ocr_result["lines"]
    print(f"[analyze] OCR result: {ocr_result['raw_text']}...")

    if not ocr_result["raw_text"].strip():
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": "No readable text could be extracted from this image. "
                         "Try a clearer, well-lit, closer photo of the text.",
            },
        )

    analysis = analyze_medicine(ocr_result["raw_text"])

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "ocr_text": ocr_result["raw_text"],
            "analysis": analysis,
        },
    )


@app.post("/api/analyze")
async def analyze_api(image: UploadFile = File(...)):
    """JSON API equivalent of /analyze, for programmatic / non-template use."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Unsupported file type. Upload JPEG, PNG, or WEBP.")

    image_bytes = await image.read()
    ocr_result = extract_text(image_bytes)
    print(f"[analyze] OCR result: {ocr_result['raw_text']}...")

    if not ocr_result["raw_text"].strip():
        return JSONResponse({"error": "No readable text extracted."}, status_code=422)

    analysis = analyze_medicine(ocr_result["raw_text"])
    return {"ocr_text": ocr_result["raw_text"], "analysis": analysis}
