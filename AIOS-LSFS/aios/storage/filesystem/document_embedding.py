from functools import cached_property
import threading

import numpy as np
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from tokenizers import Tokenizer


EMBEDDING_VERSION = "all-minilm-l6-v2-token-weighted-mean-v1"


class DocumentEmbedder:
    """Represent all document chunks with a normalized, token-weighted mean."""

    def __init__(self):
        self.model = ONNXMiniLM_L6_V2()
        self.lock = threading.RLock()

    @cached_property
    def tokenizer(self):
        # The public call ensures the model/tokenizer files are available first.
        self.model([""])
        tokenizer = Tokenizer.from_str(self.model.tokenizer.to_str())
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer

    def split_document(self, content):
        encoding = self.tokenizer.encode(content, add_special_tokens=False)
        offsets = encoding.offsets
        # Reserve space for the model's [CLS] and [SEP] tokens.
        limit = self.model.max_tokens() - self.tokenizer.num_special_tokens_to_add(False)
        chunks, weights = [], []
        start = 0
        while start < len(offsets):
            end = min(start + limit, len(offsets))
            while True:
                chunk = content[offsets[start][0]:offsets[end - 1][1]]
                weight = len(self.tokenizer.encode(chunk, add_special_tokens=False).ids)
                if weight <= limit:
                    break
                # A word-piece boundary can tokenize differently when split.
                end -= 1
                if end <= start:
                    raise ValueError("A document chunk exceeds the embedding token limit")
            chunks.append(chunk)
            weights.append(weight)
            start = end
        return chunks, weights

    def embed(self, content):
        if not isinstance(content, str):
            raise TypeError("Embedding input must be a string")
        with self.lock:
            chunks, weights = self.split_document(content)
            if not chunks:
                return self.model([""])[0].tolist()
            embeddings = np.asarray(self.model(chunks), dtype=np.float64)
            vector = np.average(embeddings, axis=0, weights=weights)
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("The document embedding has no valid direction")
            return (vector / norm).astype(np.float32).tolist()
