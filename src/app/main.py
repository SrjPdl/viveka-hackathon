from fastapi import FastAPI, UploadFile
from torchvision import transforms
from PIL import Image
import torch
from src.components.models_architecture import InceptBaseModel
from src.utils import get_latest_best_model
import os 
from fastapi import HTTPException, status

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    global device
    global preprocess
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    model_path = get_latest_best_model(os.path.join(grandparent_dir,"artifacts/model"))
    model = InceptBaseModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((229, 229)),
        transforms.ToTensor(),
    ])

@app.get("/")
def root():
    return {"message": "Fraud Document detection. Go to http://127.0.0.1:8000/docs for API documentation and testing"}

@app.post("/predict")
def predict(file: UploadFile):
    allowed_content_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image uploads are allowed (JPEG, PNG).",
        )
    image = Image.open(file.file).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
    
    return {"Prediction": output.item()}