import os
import torch
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model_architecture import SpiralModelV0

# ─────────────────────────────────────────────
#  Path Resolution
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# ─────────────────────────────────────────────
#  Startup Validation — runs before server opens
# ─────────────────────────────────────────────
print(f"📁 BASE_DIR   : {BASE_DIR}")
print(f"📁 STATIC_DIR : {STATIC_DIR}  | exists={STATIC_DIR.exists()}")
print(f"📄 index.html : {INDEX_FILE}  | exists={INDEX_FILE.exists()}")

if not STATIC_DIR.exists():
    print("❌ FATAL: 'static/' directory not found. Create it and place index.html inside.")
    os._exit(1)

if not INDEX_FILE.exists():
    print("❌ FATAL: 'static/index.html' not found. Move your HTML file into the static/ folder.")
    os._exit(1)

# ─────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Spiral Model Classification API",
    description="FastAPI serving a 3-class spiral neural network classification model.",
    version="1.0.0"
)

# ─────────────────────────────────────────────
#  Static Files Mount
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────
#  Root Route → serves index.html
# ─────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(str(INDEX_FILE), media_type="text/html")

# ─────────────────────────────────────────────
#  Debug Route → verify paths at runtime
# ─────────────────────────────────────────────
@app.get("/debug")
async def debug_paths():
    return {
        "base_dir":        str(BASE_DIR),
        "static_dir":      str(STATIC_DIR),
        "static_exists":   STATIC_DIR.exists(),
        "index_exists":    INDEX_FILE.exists(),
        "files_in_static": os.listdir(str(STATIC_DIR)) if STATIC_DIR.exists() else "DIRECTORY MISSING"
    }

# ─────────────────────────────────────────────
#  Model Setup
# ─────────────────────────────────────────────
device = torch.device("cpu")
model  = None

try:
    model = SpiralModelV0(input_size=2, hidden_dim=10, output_size=3)

    weights_path = BASE_DIR / "model_spiral.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"❌ Weights not found at: {weights_path}")

    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully on CPU with matching shapes!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    os._exit(1)

# ─────────────────────────────────────────────
#  Input Schema
# ─────────────────────────────────────────────
class SpiralInput(BaseModel):
    x1: float
    x2: float

# ─────────────────────────────────────────────
#  Prediction Endpoint
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(item: SpiralInput):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model uninitialized."})

    try:
        input_tensor = torch.tensor(
            [[item.x1, item.x2]], dtype=torch.float32, device=device
        )

        

        with torch.no_grad():
            raw_logits      = model(input_tensor)
            # 1. Name this variable 'logits_list' instead of 'probabilities'
            logits_list     = raw_logits.squeeze().tolist()
            predicted_class = torch.argmax(raw_logits, dim=1).item()

        return {
            # 2. Now 'logits_list' matches perfectly, and the yellow line will vanish!
            "prediction_logits": logits_list,  
            "predicted_class":   predicted_class
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

# ─────────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status":      "healthy",
        "model_ready": model is not None,
        "index_found": INDEX_FILE.exists()
    }