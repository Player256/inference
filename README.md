# inference

Inference experiments focused on agentic workloads.

## LLM deployment v1

A simple first version of deploying an LLM on a single GPU. The benchmark varies only request concurrency, without exploring other optimizations.

### Benchmark results

Each run completed 50 requests with no failures, using 410,120 input tokens and generating 102,400 output tokens.

| Metric | Concurrency 1 | Concurrency 4 | Concurrency 8 |
|---|---:|---:|---:|
| Benchmark duration (s) | 2191.18 | 691.56 | 623.14 |
| Request throughput (req/s) | 0.02 | 0.07 | 0.08 |
| Output token throughput (tok/s) | 46.73 | 148.07 | 164.33 |
| Peak output token throughput (tok/s) | 68.00 | 186.00 | 228.00 |
| Peak concurrent requests | 2.00 | 8.00 | 14.00 |
| Total token throughput (tok/s) | 233.90 | 741.10 | 822.47 |
| Mean TTFT (ms) | 2252.77 | 3996.35 | 7131.98 |
| Median TTFT (ms) | 2269.41 | 3944.45 | 7274.31 |
| P99 TTFT (ms) | 4245.37 | 7152.28 | 14148.48 |
| Mean TPOT (ms) | 20.31 | 24.14 | 40.50 |
| Median TPOT (ms) | 20.30 | 24.14 | 40.33 |
| P99 TPOT (ms) | 20.40 | 25.11 | 42.88 |
| Mean ITL (ms) | 20.32 | 24.15 | 40.51 |
| Median ITL (ms) | 20.25 | 23.27 | 37.53 |
| P99 ITL (ms) | 50.52 | 47.24 | 132.40 |

TTFT is time to first token, TPOT is time per output token excluding the first token, and ITL is inter-token latency.
