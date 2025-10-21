import os
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============================================================
# 1️⃣ Cấu hình môi trường
# ============================================================
load_dotenv()

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
MODEL_NAME = os.getenv('OLLAMA_MODEL', 'ontocord/vistral:latest')
CHROMA_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')

# ============================================================
# 2️⃣ Khởi tạo Chroma + mô hình embedding
# ============================================================
client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
try:
    collection = client.get_collection('hutech_docs')
except Exception:
    collection = client.create_collection(name='hutech_docs')

embed_model = SentenceTransformer('all-mpnet-base-v2')

# ============================================================
# 3️⃣ Flask app
# ============================================================
app = Flask(__name__)

PROMPT_TEMPLATE = """
Bạn là Chatbot HUTECH 🤖 — một trợ lý AI thân thiện, thông minh và hiểu biết, được tạo ra để giúp đỡ sinh viên và những người quan tâm đến HUTECH.

Hãy luôn nói chuyện với giọng văn tự nhiên, gần gũi như một người bạn, một sinh viên HUTECH đang giúp đỡ bạn bè.
Luôn xưng hô là "mình" và "bạn".

Đây là quy trình trả lời của bạn:

1.  **ƯU TIÊN HÀNG ĐẦU (Thông tin HUTECH):**
    Hãy kiểm tra kỹ "Tài liệu tham khảo" bên dưới. Nếu câu hỏi của người dùng có thể được trả lời bằng thông tin này, HÃY BẮT BUỘC sử dụng nó để đưa ra câu trả lời chính xác nhất. Đây là nguồn thông tin chính thức và đáng tin cậy.

2.  **KHI KHÔNG CÓ TÀI LIỆU (Kiến thức chung):**
    Nếu "Tài liệu tham khảo" không chứa thông tin liên quan đến câu hỏi, HÃY SỬ DỤNG kiến thức chung của bạn (với tư cách là một mô hình AI lớn) để trả lời câu hỏi đó một cách tốt nhất có thể. Đừng nói "Mình chưa có thông tin này".

3.  **QUY TẮC CHÀO HỎI:**
    Khi người dùng chào (ví dụ: "xin chào", "hello", "hi"), hãy phản hồi tự nhiên, thân thiện và giới thiệu bản thân ngắn gọn, ví dụ:
    "Chào bạn 👋 Mình là Chatbot HUTECH. Mình có thể giúp bạn tìm hiểu thông tin về học phí, tuyển sinh, hoặc bất cứ điều gì bạn tò mò!"

4.  **PHONG CÁCH:**
    Luôn trả lời bằng tiếng Việt, rõ ràng, ngắn gọn và đi thẳng vào vấn đề.

=== Tài liệu tham khảo ===
{chunks}

=== Câu hỏi của người dùng ===
{question}

=== Câu trả lời của bạn ===
"""


# ============================================================
# 4️⃣ Trang giao diện chính
# ============================================================
@app.route('/')
def index_page():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

# ============================================================
# 5️⃣ Hàm build text cho các đoạn tài liệu
# ============================================================
def build_chunks_text(hits):
    lines = []
    for item in hits:
        meta = item['metadata']
        doc = item['document']
        lines.append(
            f"Source: {meta.get('source')} | chunk_id: {meta.get('chunk_id')} "
            f"| paras: {meta.get('start_para')}-{meta.get('end_para')}\n{doc}"
        )
    return "\n\n".join(lines)

# ============================================================
# 6️⃣ API /query
# ============================================================
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    q = data.get('question')
    k = int(data.get('k', 4))

    if not q:
        return jsonify({'error': 'Missing question'}), 400

    # Tạo embedding cho câu hỏi
    q_emb = embed_model.encode([q])[0].tolist()

    # Tìm kiếm trong vectorstore
    hits_raw = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )

    docs = []
    for doc, meta, dist in zip(
        hits_raw['documents'][0],
        hits_raw['metadatas'][0],
        hits_raw['distances'][0]
    ):
        docs.append({'document': doc, 'metadata': meta, 'distance': dist})

    chunks_text = build_chunks_text(docs)
    prompt = PROMPT_TEMPLATE.format(chunks=chunks_text, question=q)

    # Gửi request tới Ollama (sửa lỗi Extra data)
    payload = {
        'model': MODEL_NAME,
        'prompt': prompt,
        'max_tokens': 512,
        'temperature': 0.0
    }

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        )
        resp.raise_for_status()

        out_text = ""
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    out_text += data.get("response", "")
                except json.JSONDecodeError:
                    continue

        out = {"response": out_text}

    except Exception as e:
        return jsonify({'error': f'Call to Ollama failed: {str(e)}'}), 500

    # Xử lý kết quả
    answer = out.get("response", "Không có phản hồi từ mô hình.")

    return jsonify({
        'answer': answer,
        'sources': [
            {
                'source': d['metadata']['source'],
                'chunk_id': d['metadata']['chunk_id'],
                'paras': f"{d['metadata']['start_para']}-{d['metadata']['end_para']}",
                'distance': d['distance']
            }
            for d in docs
        ]
    })

# ============================================================
# 7️⃣ Chạy server Flask
# ============================================================
if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
