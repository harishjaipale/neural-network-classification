import os
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from model_architecture import SpiralModelV0

# ✅ FIX: Resolve absolute paths relative to this file's location
#         This ensures routes and static files work regardless of
#         which directory Uvicorn is launched from.
BASE_DIR  = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Spiral Model Classification API",
    description="FastAPI serving a 3-class spiral neural network classification model.",
    version="1.0.0"
)

# --- Serve Frontend UI ---
# Mount /static using the resolved absolute path
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Root route — serve index.html with existence guard
@app.get("/")
async def read_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"index.html not found. Expected at: {index_path}"}
        )
    return FileResponse(str(index_path))


# --- Machine Learning Model Setup ---
device = torch.device("cpu")
model  = None

try:
    model = SpiralModelV0(input_size=2, hidden_dim=10, output_size=3)

    weights_path = BASE_DIR / "model_spiral.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully on CPU with matching shapes!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    os._exit(1)


# --- Input/Output Schema ---
class SpiralInput(BaseModel):
    x1: float
    x2: float


# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(item: SpiralInput):
    """
    Accepts 2D spatial coordinates and returns the model prediction and predicted class.
    """
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model uninitialized."})

    try:
        input_tensor = torch.tensor([[item.x1, item.x2]], dtype=torch.float32, device=device)

        with torch.no_grad():
            raw_logits    = model(input_tensor)
            probabilities = raw_logits.squeeze().tolist()
            predicted_class = torch.argmax(raw_logits, dim=1).item()

        return {
            "prediction_logits": probabilities,
            "predicted_class":   predicted_class
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_ready": model is not None}