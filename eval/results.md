# Evaluation results

| Run | Variant | Fact recall | Faithfulness | Abstention | Context recall | Source accuracy | Latency (s, median) | Tokens / question |
|---|---|---|---|---|---|---|---|---|
| graph_v2 | graph | 75% | 87% | 100% | 0% | 29% | 9.9 | 12495 |
| hybrid_rerank | hybrid | 100% | 100% | 100% | 75% | 65% | 5.0 | 8987 |
| hybrid_v2 | hybrid | 98% | 100% | 100% | 56% | 71% | 4.6 | 8976 |
| vector_v2 | vector | 91% | 99% | 100% | 50% | 35% | 3.3 | 7673 |
