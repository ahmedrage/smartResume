# Smart Resume Analyzer Backend

Welcome to the Smart Resume Analyzer backend, a powerful tool for predicting skills and roles from resume text using a convolutional neural network (CNN) based multi-label classification approach. This project is based on the research paper:

> Jiechieu, K.F.F., Tsopze, N. Skills prediction based on multi-label resume classification using CNN with model predictions explanation. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05302-x

You can find the live application at [https://smart-resume.meteorapp.com/](https://smart-resume.meteorapp.com/).

## Features

- **Predict Skills/Roles**: Given a resume text, the API predicts various skills and roles using a pre-trained model.
- **Extract Text from Files**: Upload a resume file (PDF or DOCX), and the API extracts the text content for analysis.
- **CORS Support**: The API supports Cross-Origin Resource Sharing (CORS) to allow integration with other applications.

## Endpoints

### Root Endpoint

- **GET /**
  - **Description**: Simple endpoint to check if the API is running.
  - **Response**: `{ "Hello": "World" }`

### Predict Skills/Roles

- **POST /predict**
  - **Description**: Predicts the skills and roles based on the provided resume text.
  - **Request Body**: 
    - `text` (string): The resume text to be analyzed.
  - **Response**:
    - A dictionary where keys are skill/role labels and values are the predicted probabilities.

### Extract Text from File

- **POST /extract-text**
  - **Description**: Extracts text content from an uploaded resume file (PDF or DOCX).
  - **Request**:
    - `file` (UploadFile): The resume file to extract text from.
  - **Response**:
    - `text` (string): The extracted text content from the uploaded file.

## How to Run

### Prerequisites

- Python 3.7+
- FastAPI
- Uvicorn
- pdfminer.six
- python-docx
- pydantic

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository/smart-resume-analyzer-backend.git
    cd smart-resume-analyzer-backend
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    uvicorn main:app --reload
    ```

4. Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the automatically generated API documentation and interact with the API.

## Project Structure

```
├── app
│ ├── init.py
│ ├── constants.py
│ ├── model_factory.py
│ └── preprocess.py
├── main.py
├── requirements.txt
└── README.md
```

- `main.py`: The main FastAPI application file.
- `app/`: Directory containing the core application modules.
  - `constants.py`: File containing constants used in the application.
  - `model_factory.py`: Module to build and load the prediction models.
  - `preprocess.py`: Module for text preprocessing.

## References

- **Research Paper**: Jiechieu, K.F.F., Tsopze, N. Skills prediction based on multi-label resume classification using CNN with model predictions explanation. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05302-x

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please feel free to contact us at [00.ahmed.azhar@gmail.com](mailto:00.ahmed.azhar@gmail.com).

---

Thank you for using Smart Resume Analyzer!
