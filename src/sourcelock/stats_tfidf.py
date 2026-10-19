    """Tiny TF-IDF vector space: sublinear TF, smoothed IDF, cosine scoring."""

    def __init__(self, docs: list[list[str]]) -> None:
        self.n = len(docs)
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf = {t: math.log((1 + self.n) / (1 + c)) + 1.0 for t, c in df.items()}

    def vector(self, tokens: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        return {t: (1.0 + math.log(c)) * self.idf[t] for t, c in tf.items() if t in self.idf}

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# MinHash LSH (near-duplicate search at scale)
# ---------------------------------------------------------------------------

def _minhash_seed(seed: int, shingle: str) -> int:
    digest = hashlib.sha256(f"{seed}:{shingle}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def minhash_signature(shingles: set[str], num_perm: int = 64) -> list[int]:
    """Deterministic MinHash signature (sha256-seeded permutations)."""
    if not shingles:
