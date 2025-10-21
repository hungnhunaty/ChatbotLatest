# HUTECH Chatbot — Vistral (ontocord/vistral) + Ollama + Chroma (Local RAG)

**Mục tiêu:** prototype RAG trả lời câu hỏi sinh viên HUTECH bằng dữ liệu trong file `.docx`.
Model chạy local via **Ollama**: `ontocord/vistral:latest` (không kèm model weights trong repo).

## Yêu cầu
- Windows (PowerShell) — hướng dẫn dùng PowerShell trong README.
- Python 3.9+
- Ollama đã cài và đã `ollama pull ontocord/vistral:latest`.
- RAM: >= 8GB recommended for Vistral-7B; nếu máy yếu, dùng model nhẹ hơn.

## Cấu trúc
```
hutech_chatbot_vistral/
  data/                    # sample sanitized .docx (included)
  src/
    ingest_docx.py         # ingest .docx -> chunk -> embeddings -> Chroma
    app_ollama.py          # Flask API: query -> retrieval -> call Ollama REST
    sanitize_sample.py     # simple sanitizer for demo
  .env.example
  .gitignore
  requirements.txt
  README.md
  Dockerfile
  run_windows.ps1          # helper script for Windows (PowerShell)
```

## Hướng dẫn chạy (Windows, PowerShell)

1. Mở PowerShell, chuyển tới thư mục chứa file `hutech_chatbot_vistral.zip` và giải nén (hoặc clone repo nếu trên Git).
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
6. (Tùy chọn) sanitize dữ liệu gốc trước khi ingest:
   ```powershell
   python src\sanitize_sample.py data\raw_hutech.docx data\sample_hutech.docx
   ```
   Trong archive mình đã kèm `data/sample_hutech.docx` để test.
7. Ingest file .docx vào Chroma:
   ```powershell
   python src\ingest_docx.py --input data\sample_hutech.docx --persist ./chroma_db
   ```
   Thao tác này sẽ tạo thư mục `chroma_db/` (KHÔNG commit thư mục này lên Git).
8. Chạy Flask API:
   ```powershell
   $env:OLLAMA_HOST='http://localhost:11434'
   $env:OLLAMA_MODEL='ontocord/vistral:latest'
   $env:CHROMA_PERSIST_DIR='./chroma_db'
   python src\app_ollama.py
   ```
9. Gửi truy vấn thử (PowerShell curl):
   ```powershell
   Invoke-RestMethod -Uri http://127.0.0.1:7860/query -Method POST -ContentType 'application/json' -Body (@{question='Lịch thi học kỳ 1 2025 là gì?'} | ConvertTo-Json)
   ```

## Ghi chú quan trọng
- **Không** commit model weights (`models/`) hoặc `chroma_db/` vào GitHub.
- Nếu Ollama trả lỗi OOM khi load model, cân nhắc chọn model nhẹ hơn hoặc tăng RAM/swap.
- Kiểm tra logs của Ollama (cửa sổ nơi bạn chạy `ollama serve`) để debug lỗi gọi REST.

## Liên hệ / Tùy chỉnh
- Nếu bạn muốn mình thêm UI web đơn giản (HTML/JS) hoặc React widget, mình có thể bổ sung.
- Nếu muốn chuyển sang llama.cpp/gpt4all thay vì Ollama, mình hỗ trợ chuyển đổi mã.
