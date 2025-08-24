import re
import os
from typing import List, Tuple, Dict

try:
    import tiktoken
    TOKEN_ENCODER_AVAILABLE = True
except Exception:
    TOKEN_ENCODER_AVAILABLE = False

EXT_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".php": "php",
    ".rb": "ruby",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".sh": "shell",
    ".ps1": "powershell",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}

SPLIT_PATTERNS = {
    "python": re.compile(r'(?=(^def\s+|^class\s+))', re.MULTILINE),
    "javascript": re.compile(r'(?=(^function\s+|^class\s+|^[a-zA-Z0-9_]+\s*=\s*function\s*\(|^[a-zA-Z0-9_]+\s*:\s*function\s*\())', re.MULTILINE),
    "typescript": re.compile(r'(?=(^function\s+|^class\s+|^interface\s+|^export\s+class|^[a-zA-Z0-9_]+\s*:\s*\())', re.MULTILINE),
    "java": re.compile(r'(?=(^\s*(public|private|protected)?\s*(class|interface|enum)\s+|^\s*(public|private|protected)\s+[^\n]+\s*\())', re.MULTILINE),
    "go": re.compile(r'(?=(^func\s+|^type\s+))', re.MULTILINE),
    "c": re.compile(r'(?=(^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(|^\s*(struct|enum|typedef)\s+))', re.MULTILINE),
    "cpp": re.compile(r'(?=(^\s*template<|^\s*(class|struct)\s+|^\s*[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\())', re.MULTILINE),
    "php": re.compile(r'(?=(^function\s+|^class\s+|^\s*namespace\s+))', re.MULTILINE),
    "ruby": re.compile(r'(?=(^def\s+|^class\s+|^module\s+))', re.MULTILINE),
}

class CodePreprocessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer, self.embed_model, self.device = self._load_embedder(model_name)

    def _load_embedder(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        embed_model = AutoModel.from_pretrained(model_name).to(device)
        return tokenizer, embed_model, device

    def _detect_language_from_path(self, path: str, text_sample: str = "") -> str:
        _, ext = os.path.splitext(path.lower())
        if ext in EXT_LANG_MAP:
            return EXT_LANG_MAP[ext]
        if text_sample.lstrip().startswith("#!") and "python" in text_sample[:100]:
            return "python"
        if "import java" in text_sample or "class " in text_sample and ";" in text_sample:
            return "java"
        if "package " in text_sample and "func " in text_sample:
            return "go"
        return "text"

    def _normalize_text(self, s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = s.replace("\x00", "")
        s = re.sub(r'[A-Za-z0-9+/]{200,}={0,2}', '<LONG_BASE64_REMOVED>', s)
        s = "\n".join(line.rstrip() for line in s.splitlines())
        return s

    def _split_by_language_chunks(self, text: str, language: str) -> List[str]:
        pattern = SPLIT_PATTERNS.get(language)
        if pattern:
            raw_parts = re.split(pattern, text)
            chunks = [p.strip() for p in raw_parts if p and p.strip()]
            if len(chunks) >= 2:
                return chunks
        para_chunks = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        avg_len = sum(len(c) for c in para_chunks) / (len(para_chunks) or 1)
        if para_chunks and avg_len > 200:
            return para_chunks
        lines = text.splitlines()
        if not lines:
            return []
        window = 200
        out = []
        for i in range(0, len(lines), window):
            out.append("\n".join(lines[i:i+window]).strip())
        return out

    def _get_token_count(self, text: str, model: str = "gpt-o3-mini") -> int:
        if TOKEN_ENCODER_AVAILABLE:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                try:
                    enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    enc = None
            if enc:
                return len(enc.encode(text))
        return max(1, len(text) // 4)

    def _chunk_by_tokens(self, text: str, max_tokens: int = 200, overlap: int = 100, model: str = "gpt-o3-mini") -> List[Tuple[str, int]]:
        if not text:
            return []
        if TOKEN_ENCODER_AVAILABLE:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                try:
                    enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    enc = None
            if enc:
                tokens = enc.encode(text)
                chunks = []
                i = 0
                N = len(tokens)
                while i < N:
                    slice_tokens = tokens[i:i+max_tokens]
                    chunk_text = enc.decode(slice_tokens)
                    chunks.append((chunk_text, len(slice_tokens)))
                    i += max_tokens - overlap
                return chunks

        approx_char_limit = max(50, max_tokens * 4)
        approx_overlap = overlap * 4
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = min(L, start + approx_char_limit)
            chunk_text = text[start:end]
            chunks.append((chunk_text, self._get_token_count(chunk_text, model)))
            start = end - approx_overlap
            if start < 0:
                start = 0
        return chunks

    def preprocess(self, text_data: List[str], file_paths: List[str], max_tokens: int = 200, overlap: int = 100, model: str = "gpt-4o-mini") -> List[Dict]:
        documents = []
        for idx, (text, path) in enumerate(zip(text_data, file_paths)):
            if not text or not isinstance(text, str):
                continue
            text = self._normalize_text(text)
            lang = self._detect_language_from_path(path, text[:1000])
            blocks = self._split_by_language_chunks(text, lang)
            if not blocks:
                continue
            chunk_id = 0
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                chunks = self._chunk_by_tokens(block, max_tokens=max_tokens, overlap=overlap, model=model)
                for chunk_text, tok_count in chunks:
                    if not chunk_text.strip():
                        continue
                    doc = {
                        "path": path,
                        "language": lang,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "tokens": tok_count,
                    }
                    documents.append(doc)
                    chunk_id += 1
        return documents

    def get_embedding(self, text: str):
        import torch
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.embed_model(**inputs).last_hidden_state
        return output.mean(dim=1).cpu().numpy()

    def embed_documents(self, documents: List[Dict]):
        from tqdm import tqdm
        embeddings = []
        for doc in tqdm(documents):
            embeddings.append(self.get_embedding(doc['text']))
        return embeddings
