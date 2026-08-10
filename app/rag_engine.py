# -*- coding: utf-8 -*-
"""
rag_engine.py
AtoCatch RAG 엔진 - Supabase pgvector 기반
"""

import os
import re
import requests

# ── 상수 ──────────────────────────────────────────────
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(_BASE_DIR, "data")
EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 4


# ── Supabase 클라이언트 (secret 키 — RLS 우회, 관리자용 인덱싱 작업 전용) ──
def _get_supabase():
    try:
        import streamlit as st
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_SECRET_KEY"]
    except Exception:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SECRET_KEY", "")

    from supabase import create_client
    return create_client(url, key)


# ── 임베딩 호출 ───────────────────────────────────────
def _embed(texts: list, api_key: str) -> list:
    resp = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


# ── 텍스트 청킹 ───────────────────────────────────────
def _chunk(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 30]


# ── PDF 텍스트 추출 ───────────────────────────────────
def _extract_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf, io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        raise RuntimeError(f"PDF 추출 실패: {e}")


# ── 문서 인덱싱 ───────────────────────────────────────
def index_document(file_bytes: bytes, filename: str, api_key: str) -> int:
    if filename.lower().endswith(".pdf"):
        text = _extract_pdf(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    chunks = _chunk(text)
    if not chunks:
        return 0

    sb = _get_supabase()

    # 같은 파일 기존 데이터 삭제
    sb.table("rag_documents").delete().eq("source", filename).execute()

    # 배치 임베딩 (100개씩)
    embeddings = []
    for i in range(0, len(chunks), 100):
        embeddings.extend(_embed(chunks[i:i+100], api_key))

    # Supabase에 upsert
    rows = [
        {
            "id": f"{filename}_{i}",
            "doc": chunk,
            "embedding": emb,
            "source": filename,
            "chunk_index": i
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    # 100개씩 배치 insert
    for i in range(0, len(rows), 100):
        sb.table("rag_documents").upsert(rows[i:i+100]).execute()

    return len(chunks)


# ── 앱 시작 시 data/ 폴더 자동 인덱싱 ────────────────
def auto_index_data_folder(api_key: str) -> dict:
    if not api_key:
        return {"indexed": [], "skipped": [], "errors": ["API 키 없음"]}

    os.makedirs(DATA_DIR, exist_ok=True)

    sb = _get_supabase()

    # 이미 인덱싱된 소스 목록 조회
    res = sb.table("rag_documents").select("source").execute()
    indexed_sources = {row["source"] for row in (res.data or [])}

    result = {"indexed": [], "skipped": [], "errors": []}

    for fname in os.listdir(DATA_DIR):
        if not (fname.lower().endswith(".pdf") or fname.lower().endswith(".txt")):
            continue

        if fname in indexed_sources:
            result["skipped"].append(fname)
            continue

        fpath = os.path.join(DATA_DIR, fname)
        try:
            with open(fpath, "rb") as f:
                file_bytes = f.read()
            n = index_document(file_bytes, fname, api_key)
            result["indexed"].append(f"{fname} ({n}청크)")
        except Exception as e:
            result["errors"].append(f"{fname}: {e}")

    return result


# ── RAG 검색 ─────────────────────────────────────────
def retrieve(query: str, api_key: str, top_k: int = TOP_K) -> list:
    sb = _get_supabase()

    q_emb = _embed([query], api_key)[0]

    # Supabase pgvector 코사인 유사도 검색
    res = sb.rpc(
        "match_rag_documents",
        {
            "query_embedding": q_emb,
            "match_count": top_k
        }
    ).execute()

    return [
        {
            "text":   row["doc"],
            "source": row["source"],
            "score":  row["similarity"]
        }
        for row in (res.data or [])
    ]


# ── 통계 ─────────────────────────────────────────────
def get_stats(api_key: str = "") -> dict:
    sb = _get_supabase()
    res = sb.table("rag_documents").select("source").execute()
    sources = sorted({row["source"] for row in (res.data or [])})
    return {"total_chunks": len(res.data or []), "sources": sources}


# ── 문서 삭제 ─────────────────────────────────────────
def delete_document(filename: str, api_key: str = "") -> bool:
    sb = _get_supabase()
    res = sb.table("rag_documents").delete().eq("source", filename).execute()
    return len(res.data or []) > 0