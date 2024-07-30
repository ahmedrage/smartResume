from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.preprocess import PreProcessor
from app.model import ResumeCNNModel
from pdfminer.high_level import extract_text as extract_text_from_pdf
from docx import Document
import json
import tempfile
import os

app = FastAPI()

# Set up CORS
origins = [
    "http://localhost:3000",
    "https://smart-resume.meteorapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load models and other heavy initializations in the global scope if necessary
pre_processor = PreProcessor()
r_model = ResumeCNNModel()
model = r_model.build_model()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        text = body.get('text', '')

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Preprocess the text
        matrix = pre_processor.text_to_matrix(text)

        # Reshape the matrix as expected by the model
        expected_shape = (1, pre_processor.cv_length, pre_processor.ncol)
        reshaped_m = matrix.reshape(expected_shape)

        # Make predictions
        pred_dic = {}
        for label in r_model.labels:
            modelx = model[label]
            predictions = modelx.predict(reshaped_m)
            pred_dic[label] = float(predictions[0][0])

        return pred_dic
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-text")
async def extract_text_from_file(file: UploadFile = File(...)):
    try:
        # Determine the file extension
        file_ext = os.path.splitext(file.filename)[1].lower()

        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Extract text based on file type
        if file_ext == ".pdf":
            text = extract_text_from_pdf(temp_file_path)
        elif file_ext in [".doc", ".docx"]:
            doc = Document(temp_file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Remove the temporary file
        os.remove(temp_file_path)

        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from file")

        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
