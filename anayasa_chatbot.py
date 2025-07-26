import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def clean_text(text, max_length=100):
    """Metni temizleyip kısaltır"""
    text = ' '.join(text.split())  # Fazla boşlukları temizle
    return text[:max_length] + '...' if len(text) > max_length else text

# API Anahtarını yükle
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF Yükleme
pdf_path = "anayasa.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Metni bölme
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(pages)

# Embedding ve vektör veritabanı
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.from_documents(texts, embeddings)
db.save_local("anayasa_index")

# QA zinciri oluşturma
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Kullanıcı arayüzü
print("\n" + "="*50)
print("🇹🇷 ANAYASA BİLGİ SİSTEMİ".center(50))
print("="*50)
print("\nMerhaba! Anayasa ile ilgili sorularınızı yanıtlayabilirim.")
print("Çıkmak için 'çıkış' yazabilirsiniz.\n")

while True:
    try:
        query = input("\n💬 Sorunuz: ")
        if query.lower() in ['exit', 'çıkış', 'quit']:
            print("\nİyi günler dilerim!")
            break
        
        result = qa({"query": query})
        
        # Formatlı çıktı
        print("\n" + "▌"*50)
        print(f"🔍 SORU: {query}")
        print("▌"*50)
        print(f"\n📝 CEVAP: {result['result']}")
        
        # Kaynak bilgileri
        print("\n📚 İLGİLİ KAYNAKLAR:")
        for i, doc in enumerate(result['source_documents'], 1):
            page_num = doc.metadata['page'] + 1
            cleaned_content = clean_text(doc.page_content)
            print(f"{i}. [Sayfa {page_num}] {cleaned_content}")
        print("▌"*50 + "\n")
        
    except Exception as e:
        print(f"\n⚠️ Hata oluştu: {str(e)}")
        print("Lütfen tekrar deneyin.\n")