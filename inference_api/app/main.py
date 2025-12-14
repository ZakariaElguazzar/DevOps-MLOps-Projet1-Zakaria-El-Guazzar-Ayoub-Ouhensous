from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import tensorflow as tf

from app.utils.preprocessing import preprocess_cnn, preprocess_mobilenet
from app.utils.labels import SELECTED_CLASSES

app = FastAPI(title="Image Classifier UI")

# Templates & static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load models once
cnn_model = tf.keras.models.load_model("../models/CNN_simple.keras")
mobilenet_model = tf.keras.models.load_model("../models/MobileNetV2_ft.keras")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict-ui")
async def predict_ui(
    request: Request,
    model_name: str = Query(...),
    file: UploadFile = File(...)
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    if model_name == "cnn":
        x = preprocess_cnn(image)
        model = cnn_model
    else:
        x = preprocess_mobilenet(image)
        model = mobilenet_model

    preds = model.predict(x)[0]
    class_id = int(preds.argmax())

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": SELECTED_CLASSES[class_id],
            "confidence": f"{preds[class_id]*100:.2f}%",
            "model": model_name
        }
    )

