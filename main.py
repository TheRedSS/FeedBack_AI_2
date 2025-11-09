import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import shutil
from typing import List, Optional

from pdf_extractor import PDFExtractor
from vector_db import VectorDatabase
from summarizer import TextSummarizer

# .env
load_dotenv()

app = FastAPI(title="PDF İşleme ve Akıllı Özetleme API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pdf_extractor = PDFExtractor(ocr_engine="tesseract") 
vector_db = VectorDatabase()
summarizer = TextSummarizer()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF dosyası yükle ve işle"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları kabul edilir")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        extracted_text = pdf_extractor.extract_text_from_pdf(file_path)
        
        # Vektör veritabanına ekle
        doc_id = file.filename.replace('.pdf', '')
        chunks_added = vector_db.add_document(
            doc_id=doc_id,
            text=extracted_text,
            metadata={"filename": file.filename}
        )
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "doc_id": doc_id,
            "chunks_added": chunks_added,
            "text_length": len(extracted_text)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")
    
    finally:
        # Geçici dosyayı temizle
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/process-directory/")
async def process_directory(directory_path: str = Form(...)):
    """Belirtilen dizindeki tüm PDF'leri işle"""
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=400, detail="Dizin bulunamadı")
    
    try:
        results = pdf_extractor.process_directory(directory_path)
        
        for filename, text in results.items():
            doc_id = filename.replace('.pdf', '')
            vector_db.add_document(
                doc_id=doc_id,
                text=text,
                metadata={"filename": filename}
            )
        
        return JSONResponse({
            "status": "success",
            "processed_files": list(results.keys()),
            "file_count": len(results)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """Veritabanındaki tüm belgeleri listele"""
    try:
        doc_ids = vector_db.get_document_ids()
        return JSONResponse({
            "status": "success",
            "document_count": len(doc_ids),
            "documents": doc_ids
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")

@app.post("/search/")
async def search_documents(query: str = Form(...), n_results: int = Form(5)):
    """Belgelerde arama yap"""
    try:
        results = vector_db.search(query, n_results=n_results)
        
        return JSONResponse({
            "status": "success",
            "query": query,
            "results": results
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")

@app.post("/summarize/")
async def summarize_text(text: str = Form(...), method: str = Form("openai")):
    """Metni özetle"""
    try:
        if method == "openai":
            summary = summarizer.summarize_with_openai(text)
        elif method == "langchain":
            summary = summarizer.summarize_with_langchain(text)
        else:
            raise HTTPException(status_code=400, detail="Geçersiz özetleme metodu")
        
        return JSONResponse({
            "status": "success",
            "method": method,
            "summary": summary
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")

@app.post("/ask/")
async def ask_question(question: str = Form(...), doc_id: Optional[str] = Form(None)):
    """Belge veya belgelere soru sor"""
    try:
        # Eğer belirli bir belge belirtilmişse, o belgeyi ara
        if doc_id:
            results = vector_db.search(question, n_results=3)
            context = "\n".join(results["documents"][0])
        else:
            # Tüm belgelerde ara
            results = vector_db.search(question, n_results=5)
            context = "\n".join(results["documents"][0])
        
        answer = summarizer.answer_question(question, context)
        
        return JSONResponse({
            "status": "success",
            "question": question,
            "answer": answer,
            "sources": results["metadatas"][0] if "metadatas" in results else []
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 