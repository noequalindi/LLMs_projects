import os
import torch
import gradio as gr
import fitz  # PyMuPDF
from groq import Groq
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec
from textwrap import wrap

# -------------------------
# Load environment variables
# -------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
dimension = 384 

CVS_ROUTE = os.path.join(os.path.dirname(__file__), "cvs")
if not os.path.exists(CVS_ROUTE):
    os.makedirs(CVS_ROUTE)

# check if the environment variables are set
assert PINECONE_API_KEY, "Falta definir PINECONE_API_KEY en el entorno"
assert GROQ_API_KEY, "Falta definir GROQ_API_KEY en el entorno"

# -------------------------
# Load embeddings model and pipeline
# -------------------------
# Using the model from sentence-transformers library, which is multilingual (to support Spanish)
# and has a dimension of 384.
# This model is used to generate embeddings for the CVs and the questions.
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_pipeline = pipeline("feature-extraction", model=model_id, tokenizer=model_id)

def get_embedding(text: str):
    embeddings = embedding_pipeline(text, truncation=True, max_length=dimension)
    return torch.tensor(embeddings).mean(dim=1)[0].numpy()

# -------------------------
# Initialize Pinecone 3.x
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cv-index-384"

# Crear índice si no existe
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Índice '{index_name}' creado.")
else:
    print(f"ℹ️ Índice '{index_name}' ya existe.")

# connect to the index
index = pc.Index(index_name)

# -------------------------
# Extract text from PDFs
# -------------------------
def extract_text_from_pdf(pdf_route):
    doc = fitz.open(pdf_route)
    text = ""
    for pagina in doc:
        text += pagina.get_text()
    return text.strip()

def chunk_text(text, max_tokens=dimension):
    return wrap(text, width=max_tokens, break_long_words=False, replace_whitespace=False)

# -------------------------
# Load CVs (PDF or TXT) to Pinecone
# -------------------------
def load_cvs_to_pinecone(folder="cvs"):
    print(f"📂 Procesando archivos de {folder}/...")

    for file_name in os.listdir(folder):
        if file_name.endswith(".txt") or file_name.endswith(".pdf"):
            path = os.path.join(folder, file_name)
            print(f"📄 Procesando: {file_name}")

            try:
                if file_name.endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                else:
                    content = extract_text_from_pdf(path)

                chunks = chunk_text(content, max_tokens=dimension)
                doc_id_base = file_name.replace(".txt", "").replace(".pdf", "")
                vectors = []

                for i, chunk in enumerate(chunks):
                    emb = get_embedding(chunk)
                    chunk_id = f"{doc_id_base}_chunk{i}"
                    vectors.append((chunk_id, emb.tolist(), {"text": chunk}))

                if vectors:
                    index.upsert(vectors=vectors)
                    print(f"✔️ {len(vectors)} chunks cargados para {file_name}")
                else:
                    print(f"⚠️ {file_name} está vacío o no se pudo dividir.")

            except Exception as e:
                print(f"❌ Error al procesar {file_name}: {e}")


# -------------------------
# Chatbot function
# -------------------------
def answer(pregunta):
    try:
        # obtain vector embedding for the question
        query_vector = get_embedding(pregunta)

        # search for the most similar CV in the index
        results = index.query(vector=query_vector.tolist(), top_k=1, include_metadata=True)
        matches = results.get("matches", [])

        if not matches:
            return "⚠️ No se encontró ningún CV relevante para esta pregunta."

        context = matches[0]["metadata"]["text"]
        id_cv = matches[0]["id"]

        prompt = f"Contexto del CV ({id_cv}):\n{context}\n\nPregunta: {pregunta}\nRespuesta:"

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


# -------------------------
# Load CVs to Pinecone and start Gradio app
# -------------------------

chat_history = []

def answer_chat_interface(message, history):
    response = answer(message)  
    history.append((message, response))
    return "", history


if __name__ == "__main__":
  load_cvs_to_pinecone(CVS_ROUTE)
  with gr.Blocks(css="""
  .gradio-container {background-color: #0000}
  .chatbot {height: 500px; overflow-y: auto}
  .gr-textbox textarea {height: 80px !important}
""") as demo:
    gr.Markdown("## 🤖 Chatbot sobre CVs\nHacé una pregunta sobre los CVs cargados y recibí una respuesta.")

    chatbot = gr.Chatbot(elem_id="chatbot", label="Conversación", bubble_full_width=False)
    with gr.Row():
        with gr.Column(scale=10):
            input_box = gr.Textbox(placeholder="Escribí tu pregunta acá...", label="Tu mensaje")
        with gr.Column(scale=1):
            send_btn = gr.Button("Enviar")

    state = gr.State([])

    send_btn.click(fn=answer_chat_interface, 
                    inputs=[input_box, state], 
                    outputs=[input_box, chatbot], 
                    api_name="chat")

    input_box.submit(fn=answer_chat_interface, 
                      inputs=[input_box, state], 
                      outputs=[input_box, chatbot])

  demo.launch()
