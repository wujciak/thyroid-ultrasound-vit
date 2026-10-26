import uvicorn
from fastapi import FastAPI, File, UploadFile
from model.inference import load_model, predict_image
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pred = predict_image(model, image_bytes)
    return {"prediction": pred}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
