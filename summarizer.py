import os
from openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI

class TextSummarizer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    
    def summarize_with_openai(self, text, max_tokens=1000):
        """OpenAI API ile metin özetleme"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "Sen profesyonel bir metin özetleyicisin. Verilen metni ana noktaları koruyarak özetle."},
                {"role": "user", "content": f"Aşağıdaki metni özetle:\n\n{text}"}
            ],
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def summarize_with_langchain(self, text):
        """LangChain ile metin özetleme"""
        docs = [Document(page_content=text)]
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.run(docs)
        return summary
    
    def answer_question(self, question, context):
        """Belirli bir bağlam içinde soruya cevap ver"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "Verilen bağlam içindeki bilgilere dayanarak soruları yanıtla. Eğer cevap bağlamda yoksa, bilmediğini belirt."},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content 