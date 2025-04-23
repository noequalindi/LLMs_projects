import os
import torch
import gradio as gr
import fitz  # PyMuPDF
from groq import Groq
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec
from textwrap import wrap
import re
import pandas as pd

# -------------------------
# Load environment variables
# -------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
dimension = 384 

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CVS_ROUTE = os.path.join(PROJECT_ROOT, "..", "cvs")

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
# Chatbot multiagent functions
# -------------------------

AGENTS_IDS = [f.replace(".pdf", "").replace(".txt", "").lower() for f in os.listdir(CVS_ROUTE) if f.endswith(".pdf") or f.endswith(".txt")]
DEFAULT_AGENT = next((a for a in AGENTS_IDS if "noelia" in a), AGENTS_IDS[0])

def sanitize_id(name):
    return re.sub(r"[^a-z0-9-]", "-", name.lower())

def init_indexes():
    total_agents = 0
    total_chunks = 0
    for agent_idx in AGENTS_IDS:
        file_path_txt = os.path.join(CVS_ROUTE, f"{agent_idx}.txt")
        file_path_pdf = os.path.join(CVS_ROUTE, f"{agent_idx}.pdf")
        path = file_path_pdf if os.path.exists(file_path_pdf) else file_path_txt if os.path.exists(file_path_txt) else None
        if path:
            print(f"📂 Cargando CV para agente: {agent_idx} desde {path}")
            loaded_chunks = update_agent_cv(agent_idx, path)
            total_agents += 1
            total_chunks += loaded_chunks
        else:
            print(f"⚠️ No se encontró archivo PDF o TXT para {agent_idx}")
    print(f"✅ Se cargaron {total_chunks} chunks en total para {total_agents} agentes.")      

def detect_agent_in_question(question):
    mentioned = []
    clean_question = re.sub(r"[.,;:!?()\[\]{}]", "", question.lower())
    tokens = set(clean_question.split())
    for agent_id in AGENTS_IDS:
        name_parts = agent_id.replace("_cv", "").replace("-cv", "").replace("cv", "").replace("_", " ").replace("-", " ").split()
        if any(p in tokens for p in name_parts):
            print(f"✅ Coincidencia detectada: {agent_id} por token en pregunta")
            mentioned.append(agent_id)
    return mentioned          

def create_index_by_agent(agent_id):
    agent_id = sanitize_id(agent_id)
    index_name = f"cv-index-{agent_id}"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

def delete_chunks_by_agent(agent_id):
    agent_id_sanitized = sanitize_id(agent_id)
    print(f"🧹 Eliminando chunks anteriores de {agent_id_sanitized}...")
    index = create_index_by_agent(agent_id_sanitized)
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"⚠️ No se pudo eliminar los vectores del índice {agent_id}: {e}")

def update_agent_cv(agent_id, filepath):  # returns the number of chunks loaded
    agent_id_sanitized = sanitize_id(agent_id)
    index = create_index_by_agent(agent_id_sanitized)
    delete_chunks_by_agent(agent_id_sanitized)
    if filepath.endswith(".pdf"):
        with fitz.open(filepath) as doc:
            content = "".join([page.get_text() for page in doc])
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

    chunks = chunk_text(content, max_tokens=dimension)
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        chunk_id = f"{agent_id}_chunk{i}"
        vectors.append((chunk_id, emb.tolist(), {"text": chunk}))

    index.upsert(vectors=vectors)
    print(f"✅ CV de {agent_id} actualizado con {len(vectors)} chunks.")
    return len(vectors)

def answer(question):
    try:
        print(f"📝 Pregunta recibida: {question}")
        detected_names = detect_agent_in_question(question)
        print(f"🔍 Nombres detectados con re: {detected_names}")

        if not detected_names:
            print("ℹ️ No se detectó ningún nombre, se usará el agente por defecto.")
            detected_names = [DEFAULT_AGENT]

        responses = []

        for name in detected_names:
            index = create_index_by_agent(name)
            personalized_question = f"{question} ({name})"
            query_vector = get_embedding(personalized_question)
            results = index.query(vector=query_vector.tolist(), top_k=3, include_metadata=True)
            print(f"📊 Resultados de Pinecone para {name}: {[m['id'] for m in results.get('matches', [])]}")
            matches = results.get("matches", [])

            if not matches:
                text_response = "No se encontró información relevante."
                show_name = "Noelia" if name == DEFAULT_AGENT else name.capitalize()
                responses.append(f"👤 {show_name}:\n{text_response}")
                continue

            context = "\n---\n".join([m["metadata"]["text"] for m in matches])
            # already defined in the embedding pipeline
            prompt = f"Contexto del CV de {name}:\n{context}\n\nPregunta: {personalized_question}\nRespuesta:"
            print(f"📤 Prompt enviado al modelo para {name}:\n{prompt[:500]}...\n")

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            respuesta_texto = response.choices[0].message.content.strip()
            nombre_mostrar = "Noelia" if name == DEFAULT_AGENT else name.capitalize()
            responses.append(f"👤 {nombre_mostrar}:\n{respuesta_texto}")

        return "\n\n".join(responses)

    except Exception as e:
        return f"❌ Error: {str(e)}"

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
# 🤖 Chatbot multiagente de CVs

💬 **Consejos para preguntar:**
- Si querés que respondan varios agentes, nombralos explícitamente. Ej: "¿Noelia y Patricio saben de JavaScript?"
- Si no mencionás a nadie, responderá el agente por defecto (Noelia).
- El sistema elegirá los agentes que contengan información relevante, no solo por coincidencia de nombre.
""")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Escribí tu pregunta...", label="Tu mensaje")
            state = gr.State([])
            send = gr.Button("Enviar")

            def responder_gradio(mensaje, hist):
                respuesta = answer(mensaje)
                hist.append((mensaje, respuesta))
                return "", hist

            send.click(responder_gradio, [msg, state], [msg, chatbot])
            msg.submit(responder_gradio, [msg, state], [msg, chatbot])

        with gr.Tab("Actualizar CV"):
            nombre = gr.Textbox(label="Nombre del agente (sin extensión)")
            archivo = gr.File(label="Subí un nuevo PDF o TXT")
            salida = gr.Textbox(label="Resultado")
            actualizar = gr.Button("Actualizar CV")

            def actualizar_desde_interfaz(name, file):
                agent_id = sanitize_id(name)
                filepath = file.name
                update_agent_cv(agent_id, filepath)
                return f"✅ CV de {agent_id} actualizado exitosamente."

            actualizar.click(actualizar_desde_interfaz, inputs=[nombre, archivo], outputs=salida)

        with gr.Tab("Agentes disponibles"):
            tabla = gr.Dataframe(pd.DataFrame({
                "Nombre del Agente": [a.capitalize() for a in AGENTS_IDS],
                "Índice Pinecone": [f"cv-index-{sanitize_id(a)}" for a in AGENTS_IDS]
            }), label="Tabla de Agentes", interactive=False)

            gr.Markdown("### 🔎 Ver chunks de un agente")
            selector = gr.Dropdown(choices=AGENTS_IDS, label="Elegí un agente")
            salida_chunks = gr.Textbox(label="Chunks del CV (primeros 3)", lines=10)
            btn_cargar = gr.Button("Mostrar chunks")

            def show_chunks(name):
                name = sanitize_id(name)
                idx = create_index_by_agent(name)
                response = idx.query(vector=[0.0]*dimension, top_k=3, include_metadata=True)
                texts = [m['metadata']['text'][:200] for m in response.get("matches", [])]
                return "\n---\n".join(texts) if texts else "(No hay chunks cargados o no se pueden recuperar.)"

            btn_cargar.click(fn=show_chunks, inputs=selector, outputs=salida_chunks)

    return demo

if __name__ == "__main__":
    init_indexes()
    demo = gradio_interface()
    demo.launch()
