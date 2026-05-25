"""BLEU(100) and CHRF++(100) metrics via sacrebleu."""
from sacrebleu.metrics import BLEU, CHRF

_bleu = BLEU(force=True)   # force=True: suppress detokenization warning
_chrf = CHRF(word_order=2)

def compute_metrics(predictions: list[str], references: list[str]) -> tuple[float, float]:
    """Return (BLEU, CHRF++) on 0-100 scale."""
    bleu = _bleu.corpus_score(predictions, [references]).score
    chrf = _chrf.corpus_score(predictions, [references]).score
    return bleu, chrf
