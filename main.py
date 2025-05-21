from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.parser import ResumeParser
from app.nlp_processor import NLPProcessor
from app.recommender import Recommender
from pathlib import Path
import uvicorn

app = FastAPI(title="Resume Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
resume_parser = ResumeParser()
nlp_processor = NLPProcessor()
recommender = Recommender()

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # Validate file extension
        allowed_extensions = {".pdf", ".doc", ".docx"}
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
            )

        # Read and process the resume
        content = await file.read()
        
        # Parse resume content
        parsed_data = resume_parser.parse(content, file_extension)
        
        # Process with NLP
        analyzed_data = nlp_processor.analyze(parsed_data)
        
        # Generate recommendations
        recommendations = recommender.get_recommendations(analyzed_data)
        
        return {
            "status": "success",
            "data": {
                "parsed_info": analyzed_data,
                "recommendations": recommendations
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)