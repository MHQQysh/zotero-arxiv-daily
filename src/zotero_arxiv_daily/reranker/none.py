from .base import BaseReranker, register_reranker
from ..protocol import Paper, CorpusPaper
import numpy as np


@register_reranker("none")
class NoneReranker(BaseReranker):
    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        for candidate in candidates:
            candidate.score = None
        return candidates

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        raise NotImplementedError("NoneReranker does not calculate similarity scores.")
