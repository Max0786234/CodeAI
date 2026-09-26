"""Small hybrid code retriever: BM25 + optional sentence-transformer vectors + RRF.

The lexical path works with the Python standard library. Install requirements.txt
for semantic embeddings. This keeps the project easy to run and explain.
"""
from __future__ import annotations
import math, re
from pathlib import Path
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}
TEXT_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".h", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css"}
CODE_FILE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".h", ".c", ".cs", ".php", ".rb", ".swift", ".kt"}

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def file_priority(path: str) -> float:
    lower = path.lower()
    if "readme" in lower or lower.endswith(".md"):
        return -1.5
    if Path(path).suffix.lower() in CODE_FILE_SUFFIXES:
        return 1.5
    return 0.0

def collect_chunks(root: str | Path, chunk_lines: int = 45, overlap: int = 8) -> list[dict[str, Any]]:
    root = Path(root).resolve(); chunks=[]
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or IGNORE_DIRS.intersection(path.parts):
            continue
        try: lines=path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError): continue
        step=max(1, chunk_lines-overlap)
        for start in range(0, len(lines), step):
            block=lines[start:start+chunk_lines]
            if not any(line.strip() for line in block): continue
            chunks.append({"path":str(path.relative_to(root)),"start_line":start+1,"end_line":start+len(block),"text":"\n".join(block)})
            if start+chunk_lines>=len(lines): break
    return chunks

def bm25_search(query: str, chunks: list[dict[str, Any]], top_k: int=8) -> list[dict[str, Any]]:
    q=tokenize(query)
    if not q or not chunks: return []
    docs=[]
    for c in chunks:
        path_tokens = tokenize(c["path"].replace("/"," "))
        filename = Path(c["path"]).stem.lower()
        file_hint_tokens = tokenize(filename.replace("_", " "))
        docs.append(tokenize(c["text"] + " " + c["path"].replace("/"," ") + " " + " ".join(file_hint_tokens) + " " + " ".join(path_tokens)))
    df=Counter(token for doc in docs for token in set(doc)); avgdl=sum(map(len,docs))/len(docs) or 1
    k1=1.5; b=.75; scored=[]
    for idx,doc in enumerate(docs):
        counts=Counter(doc); score=0.0
        path_text = chunks[idx]["path"].lower()
        path_tokens = set(tokenize(path_text.replace("/"," ")))
        for term in q:
            freq=counts[term]
            if not freq: continue
            idf=math.log(1+(len(docs)-df[term]+.5)/(df[term]+.5))
            score += idf*(freq*(k1+1))/(freq+k1*(1-b+b*len(doc)/avgdl))
            if term in path_tokens:
                score += 3.0
        score += file_priority(chunks[idx]["path"])
        if score>0: scored.append((score,idx))
    scored.sort(reverse=True)
    return [{**chunks[i],"score":round(s,5),"source":"bm25","rank":r} for r,(s,i) in enumerate(scored[:top_k],1)]

def embedding_search(query: str, chunks: list[dict[str, Any]], top_k: int=8, model_name: str="sentence-transformers/all-MiniLM-L6-v2") -> list[dict[str, Any]]:
    """Semantic retrieval. Model downloads on first use; requires sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Semantic search requires dependencies. Run: pip install -r requirements.txt") from exc
    if not chunks: return []
    model=SentenceTransformer(model_name)
    texts=[c["text"]+"\nFile: "+c["path"] for c in chunks]
    vectors=model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    qvec=model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores=np.asarray(vectors) @ np.asarray(qvec)
    inds=np.argsort(scores)[::-1][:top_k]
    return [{**chunks[int(i)],"score":round(float(scores[i]),5),"source":"vector","rank":r} for r,i in enumerate(inds,1)]

def reciprocal_rank_fusion(*ranked_lists: list[dict[str, Any]], k: int=60, top_k: int=8) -> list[dict[str, Any]]:
    """Combine ranked lists using RRF; identity is file path + chunk start line."""
    fused={}
    for results in ranked_lists:
        for rank,item in enumerate(results,1):
            key=(item["path"],item["start_line"])
            if key not in fused: fused[key]={**item,"rrf_score":0.0,"sources":[]}
            fused[key]["rrf_score"] += 1/(k+rank)
            if item.get("source") not in fused[key]["sources"]: fused[key]["sources"].append(item.get("source","unknown"))
    return sorted(fused.values(),key=lambda x:x["rrf_score"],reverse=True)[:top_k]

def hybrid_search(root: str | Path, query: str, top_k: int=8, semantic: bool=False) -> dict[str, Any]:
    chunks=collect_chunks(root)
    lexical=bm25_search(query,chunks,top_k=max(top_k*3,20))
    vector=[]
    if semantic: vector=embedding_search(query,chunks,top_k=max(top_k*3,20))
    fused=reciprocal_rank_fusion(lexical,vector,top_k=top_k) if semantic else lexical[:top_k]
    if not semantic:
        fused = sorted(lexical[:top_k], key=lambda x: (x.get("score", 0), file_priority(x["path"])), reverse=True)
    return {"query":query,"semantic_enabled":semantic,"chunk_count":len(chunks),"results":[{**x,"text":x["text"][:2200]} for x in fused]}
