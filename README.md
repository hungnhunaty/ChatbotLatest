# HUTECH Chatbot — Vistral (ontocord/vistral) + Ollama + Chroma (Local RAG)

**Mục tiêu:** prototype RAG trả lời câu hỏi sinh viên HUTECH bằng dữ liệu trong file `.docx`.
Model chạy local via **Ollama**: `ontocord/vistral` (không kèm model weights trong repo).

## Yêu cầu
- Windows (PowerShell) — hướng dẫn dùng PowerShell trong README.
- Python 3.9+
- Ollama đã cài và đã `ollama pull ontocord/vistral`.
- RAM: >= 8GB recommended for Vistral-7B; nếu máy yếu, dùng model nhẹ hơn.

## Cấu trúc
```
hutech_chatbot_vistral/
   data/                   
      -sample_hutech.docx
   ingest_docx.py         # ingest .docx -> chunk -> embeddings -> Chroma
   app_ollama.py          # Flask API: query -> retrieval -> call Ollama REST
   .env.example
   .gitignore
   requirements.txt
   README.md
```

## Hướng dẫn chạy (Windows, PowerShell)

1. Mở PowerShell, chuyển tới thư mục chứa file `hutech_chatbot_project` vừa giải nén (hoặc clone repo nếu trên Git).
2. Tạo virtualenv và kích hoạt:
   ```powershell
   python -m venv chatbotvenv
   .\chatbotvenv\Scripts\Activate
   ```
3. Cài Python dependencies:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Pull model Vistral về Ollama (một lần):
   ```powershell
   ollama pull ontocord/vistral:latest
   ```
5. Khởi động Ollama server (mở 1 cửa sổ PowerShell khác và để nó chạy):
   ```powershell
   ollama serve
   ```
   Kiểm tra model có sẵn:
   ```powershell
   ollama list
   ```
6. Ingest file .docx vào Chroma:
   ```powershell
   python ingest_docx.py --input data/sample_hutech.docx
   ```
   Thao tác này sẽ tạo thư mục `chroma_db/` (KHÔNG commit thư mục này lên Git).
7. Chạy Flask API:
   ```powershell
   python app_ollama.py
   ```

## Ghi chú quan trọng
- **Không** commit model weights (`models/`) hoặc `chroma_db/` vào GitHub.
- Nếu Ollama trả lỗi OOM khi load model, cân nhắc chọn model nhẹ hơn hoặc tăng RAM/swap.
- Kiểm tra logs của Ollama (cửa sổ nơi bạn chạy `ollama serve`) để debug lỗi gọi REST.
