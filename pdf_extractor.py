import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
from openai import OpenAI
import base64

class PDFExtractor:
    def __init__(self, ocr_engine="tesseract"):
        self.ocr_engine = ocr_engine
        if ocr_engine == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_text_from_pdf(self, pdf_path):
        """PDF'den metin çıkarma işlemi"""
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            
            if len(text.strip()) < 50:
                if self.ocr_engine == "tesseract":
                    text = self._apply_tesseract_ocr(page)
                elif self.ocr_engine == "openai":
                    text = self._apply_openai_ocr(page)
            
            full_text += f"\n--- Sayfa {page_num+1} ---\n{text}"
        
        return full_text
    
    def _apply_tesseract_ocr(self, page):
        """Tesseract OCR uygulama"""
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang='tur+eng')
    
    def _apply_openai_ocr(self, page):
        """OpenAI Vision API ile OCR uygulama"""
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Bu görüntüdeki tüm metni çıkar."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def process_directory(self, directory_path):
        """Belirtilen dizindeki tüm PDF'leri işle"""
        results = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                results[filename] = self.extract_text_from_pdf(file_path)
        return results 