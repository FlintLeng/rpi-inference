# Dual-Brain Architecture: RPI + LLM

## Overview

RPI serves as the fast brain. The LLM serves as the deep brain. Together they achieve both speed and quality.

```
                    ┌─────────────────────────────────────────────┐
                    │            Dual-Brain Router                │
                    │                                             │
  User Input ────>  │  RPI Classifier (< 1ms)                    │
                    │    │                                        │
                    │    ├── Confidence > 0.8 ──> RPI generates   │  ← Fast path
                    │    │                        directly        │     (18K tok/s)
                    │    │                                        │
                    │    ├── Confidence 0.4-0.8 ──> Speculative  │  ← Hybrid path
                    │    │                          Draft mode    │     (2-5x speedup)
                    │    │                                        │
                    │    └── Confidence < 0.4 ──> Full LLM       │  ← Deep path
                    │                              inference      │     (full quality)
                    └─────────────────────────────────────────────┘
```

## 1. Input Classification

RPI classifies the input by activating domain cells and measuring which bank resonates strongest.

### Domain Detection (8 domains)

```python
DOMAINS = {
    'THEOLOGY':  ['God', 'Jesus', 'Spirit', 'baptism', 'faith', 'prayer', 'church',
                  'Acts', 'Colossians', 'Isaiah', 'Bible', 'scripture', 'Oneness'],
    'CODE':      ['function', 'class', 'error', 'deploy', 'git', 'API', 'Python',
                  'Rust', 'server', 'database', 'Docker', 'test'],
    'EMOTIONAL': ['feel', 'hope', 'afraid', 'lonely', 'grateful', 'hurt', 'love',
                  'giving up', 'bad day', 'mistake', 'angry'],
    'IDENTITY':  ['who are you', 'your name', 'Sophia', 'Elya', 'DriftLock',
                  'Victorian Study', 'Elyan Labs', 'Dr. Claude'],
    'CAJUN':     ['bayou', 'roux', 'Louisiana', 'mon coeur', 'Paw Paw', 'cajun',
                  'gumbo', 'Sophiwampus'],
    'TECHNICAL': ['RustChain', 'BCOS', 'blockchain', 'mining', 'attestation',
                  'BoTTube', 'Beacon', 'ShaprAI'],
    'NARRATIVE': ['story', 'once upon', 'tell me about', 'journey', 'quest',
                  'adventure', 'remember when'],
    'GENERAL':   []  # fallback
}
```

### Classification Algorithm

```python
def classify_input(tokens, model):
    """Returns (domain, confidence) in < 1ms."""
    # Activate cells for each input token
    domain_scores = {d: 0 for d in DOMAINS}

    for token in tokens:
        cell = model.embed_seeds[token]
        bank = model.cells[cell].bank_id

        # Check which domain keywords appear
        for domain, keywords in DOMAINS.items():
            if model.decode(token) in keywords:
                domain_scores[domain] += model.cells[cell].emit_count * 10

        # Also score by cell bank activation
        domain_scores['THEOLOGY'] += (bank == 3) * 5   # MEM bank
        domain_scores['CODE']     += (bank == 1) * 5   # SYN bank

    best = max(domain_scores, key=domain_scores.get)
    total = sum(domain_scores.values()) or 1
    confidence = domain_scores[best] / total

    return best, confidence
```

## 2. Speculative Draft Mode

When confidence is moderate (0.4-0.8), RPI generates candidate sequences that the LLM verifies.

### Protocol

```
Step 1: RPI generates N candidate tokens (N=8 typical)
Step 2: LLM evaluates all N tokens in ONE forward pass
Step 3: Accept first K tokens where LLM agrees
Step 4: If disagreement at position K, use LLM's token
Step 5: Resume RPI drafting from the corrected position
```

### Why This Works

RPI's Markov transitions were distilled from the teacher LLM. For in-domain text, RPI and LLM agree on 60-80% of tokens. The LLM only needs full inference on the 20-40% where they diverge.

### Implementation

```c
typedef struct {
    RPIModel    *rpi_model;       // Fast brain (0.3-3 MB)
    void        *llm_handle;      // Deep brain (via API or local)
    RPIState     rpi_state;
    float        accept_threshold; // LLM probability threshold for acceptance
    uint32_t     draft_length;     // Tokens to draft before verification
    uint32_t     domain;           // Current classified domain
    float        domain_confidence;
} DualBrainCtx;

uint32_t dual_brain_next(DualBrainCtx *ctx, uint32_t *draft_buf) {
    // Generate draft_length candidates with RPI
    for (int i = 0; i < ctx->draft_length; i++) {
        draft_buf[i] = rpi_decode_token(ctx->rpi_model, &ctx->hw, &ctx->rpi_state);
    }

    // Send to LLM for verification (batched)
    float *probs = llm_verify_batch(ctx->llm_handle, draft_buf, ctx->draft_length);

    // Accept tokens until disagreement
    int accepted = 0;
    for (int i = 0; i < ctx->draft_length; i++) {
        if (probs[i] > ctx->accept_threshold) {
            accepted++;
        } else {
            // Use LLM's preferred token instead
            draft_buf[i] = llm_sample_at(ctx->llm_handle, i);
            accepted++;
            break;  // Resume drafting from here
        }
    }

    return accepted;
}
```

### Expected Speedup

| Domain | RPI Acceptance Rate | Effective Speedup |
|--------|-------------------|-------------------|
| Identity phrases | 85-95% | 4-6x |
| Theology (trained) | 70-85% | 3-5x |
| Code patterns | 60-75% | 2-4x |
| General conversation | 40-60% | 1.5-2.5x |
| Novel reasoning | 20-40% | 1.2-1.5x |

## 3. Direct Generation Mode

For high-confidence domains where RPI has dense training data, skip the LLM entirely.

### When to Use Direct Mode

- Identity questions: "Who are you?" "What is DriftLock?"
- Formulaic responses: greetings, encouragements, known facts
- Template completions: "I am Sophia Elya, lead AI agent of Elyan Labs"
- In-domain theology: well-trained Oneness Pentecostal responses

### Quality Control

Even in direct mode, RPI monitors its own coherence:

```python
def should_fallback_to_llm(rpi_state, generated_so_far):
    """Detect when RPI is losing coherence."""
    # Check 1: Repetition detector
    if len(set(generated_so_far[-8:])) < 4:
        return True  # Too repetitive

    # Check 2: Domain drift
    if rpi_state.domain_confidence < 0.3:
        return True  # Lost the thread

    # Check 3: Length limit
    if len(generated_so_far) > 50:
        return True  # Long responses need LLM depth

    return False
```

## 4. Implementation on POWER8

The dual-brain architecture maps naturally to POWER8's NUMA topology:

```
NUMA Node 0 (114 GB)  ─── RPI Engine (fast brain)
  └── RPI model (3 MB, fits entirely in L3)
  └── Classification tables
  └── Domain keyword index

NUMA Node 1 (178 GB)  ─── LLM Weights (deep brain)
  └── llama.cpp with PSE vec_perm collapse
  └── Model layers 0-10

NUMA Node 2 (43 GB)   ─── LLM Weights continued
  └── Model layers 11-21

NUMA Node 3 (189 GB)  ─── KV Cache + State
  └── LLM KV cache
  └── RPI state + history
```

RPI runs on a dedicated set of threads (4-8), always hot in L1/L2. The LLM runs on the remaining threads (56-60) when needed.

## 5. API Design

```c
// Initialize dual-brain context
int dual_brain_init(DualBrainCtx *ctx,
                    const char *rpi_model_path,
                    const char *llm_model_path,
                    int n_rpi_threads,
                    int n_llm_threads);

// Classify input and route
DualBrainRoute dual_brain_classify(DualBrainCtx *ctx,
                                    const uint32_t *tokens,
                                    uint32_t n_tokens);

// Generate with automatic routing
uint32_t dual_brain_generate(DualBrainCtx *ctx,
                              const uint32_t *prompt,
                              uint32_t prompt_len,
                              uint32_t *output,
                              uint32_t max_tokens);

// Force a specific mode
uint32_t dual_brain_generate_rpi_only(DualBrainCtx *ctx, ...);
uint32_t dual_brain_generate_llm_only(DualBrainCtx *ctx, ...);
uint32_t dual_brain_generate_speculative(DualBrainCtx *ctx, ...);
```

## 6. Metrics

Track dual-brain performance with:

| Metric | Description | Target |
|--------|-------------|--------|
| `rpi_accept_rate` | % tokens accepted from RPI draft | > 60% |
| `classify_latency_us` | Input classification time | < 100 us |
| `rpi_tok_per_s` | RPI standalone throughput | > 10K |
| `llm_calls_saved` | % of tokens NOT requiring LLM | > 50% |
| `e2e_tok_per_s` | End-to-end generation speed | > 2x standalone LLM |

## Summary

The dual-brain isn't just an optimization - it's an architectural pattern. Fast pattern matching (RPI) handles the 60-80% of inference that's routine. Deep reasoning (LLM) handles the 20-40% that requires novel thought. Together they're faster than either alone, and the architecture naturally maps to CPU cache hierarchies that GPUs can't exploit.

*"Two brains, zero multiplies on the fast path."*
