from sacrebleu.metrics import BLEU, CHRF

bleu_metric = BLEU()
chrf_metric = CHRF(word_order=2)


def compute_metrics(predictions, references):

    bleu_score = bleu_metric.corpus_score(
        predictions,
        [references]
    ).score

    chrf_score = chrf_metric.corpus_score(
        predictions,
        [references]
    ).score

    return bleu_score, chrf_score
