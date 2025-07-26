import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Güncel kütüphaneler
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

class HukukAsistani:
    def __init__(self):
        # Anayasa verisini yükle
        loader = PyPDFLoader("Anayasa.pdf")
        metinler = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nMadde", "\n\n"]
        ).split_documents(loader.load())
        
        self.veritabani = FAISS.from_documents(
            metinler, 
            OpenAIEmbeddings(model="text-embedding-3-large")
        )
        
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
        
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.veritabani.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template="""
                    Basit ve anlaşılır Türkçe ile yanıt ver:
                    1. Sadece Anayasa maddelerini referans al
                    2. Günlük dil kullan
                    3. Pratik sonuçları açıkla
                    
                    Soru: {question}
                    
                    İlgili Maddeler: {context}
                    
                    Yanıt:
                    """,
                    input_variables=["context", "question"]
                )
            }
        )
    
        

    def sor(self, soru):
        cevap = self.qa.invoke({"query": soru})
        return self._temizle(cevap["result"])

    def _temizle(self, metin):
        # Gereksiz İngilizce ifadeleri temizle
        temiz = metin.replace("Observation:", "").replace("Thought:", "")
        return temiz.strip()

def main():
    print("\n" + "="*50)
    print("🇹🇷 HUKUK ASİSTANI".center(50))
    print("="*50)
    print("\nAnayasa ve temel haklar konusunda sorularınızı yanıtlıyorum.")
    print("Çıkmak için 'çıkış' yazın\n")
    
    asistan = HukukAsistani()
    
    while True:
        try:
            soru = input("\n❓ Sorunuz: ")
            if soru.lower() in ['exit', 'çıkış']:
                print("\nGörüşmek üzere!")
                break
                
            print("\n" + "-"*50)
            print("🔍 Yanıt:")
            print(asistan.sor(soru))
            print("-"*50)
            
        except Exception as e:
            print(f"\n⚠️ Hata: {str(e)}")

if __name__ == "__main__":
    main()