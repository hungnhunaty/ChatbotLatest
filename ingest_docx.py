import os
import argparse
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

def load_docx(path):
    doc = Document(path)
    paras = []
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if t:
            paras.append({'text': t, 'para_idx': i})
    return paras

def chunk_paragraphs(paras, max_chars=900):
    chunks = []
    cur = ''
    meta = {'start_para': None, 'end_para': None}
    for p in paras:
        t = p['text']
        if not cur:
            cur = t
            meta = {'start_para': p['para_idx'], 'end_para': p['para_idx']}
        elif len(cur) + 1 + len(t) <= max_chars:
            cur += '\n' + t
            meta['end_para'] = p['para_idx']
        else:
            chunks.append((cur, meta.copy()))
            cur = t
            meta = {'start_para': p['para_idx'], 'end_para': p['para_idx']}
    if cur:
        chunks.append((cur, meta.copy()))
    return chunks

def main(input_path, persist_dir):
    client = PersistentClient(path=persist_dir)
    collection_name = 'hutech_docs'
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    paras = load_docx(input_path)
    chunks = chunk_paragraphs(paras, max_chars=900)
    print(f'Paragraphs: {len(paras)}, Chunks: {len(chunks)}')

    model = SentenceTransformer('all-mpnet-base-v2')
    texts = [c[0] for c in chunks]
    metas = []
    for i, (_, m) in enumerate(chunks):
        metas.append({'source': os.path.basename(input_path), 'chunk_id': str(i),
                      'start_para': m['start_para'], 'end_para': m['end_para']})

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    ids = [f'chunk-{i}' for i in range(len(texts))]
    collection.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metas, ids=ids)
    print('Ingest finished and persisted to', persist_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to .docx')
    parser.add_argument('--persist', default='./chroma_db', help='Chroma persist dir')
    args = parser.parse_args()
    main(args.input, args.persist)
