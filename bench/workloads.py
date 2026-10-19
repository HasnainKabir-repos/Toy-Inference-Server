"""
Deterministic workloads. A workload is a list of (prompt, max_new_tokens)
plus, for open-loop runs, Poisson arrival offsets. Everything is seeded so
every system sees the *identical* request stream.
 
Workloads
    uniform   short prompts, every request asks for 48 new tokens
    mixed     short prompts, output budgets drawn from a skewed mix (8..128)
    longctx   ~200-token context + question, mixed output budgets
              (stresses KV size, hence the cost of rebuilding the batched cache)
 
Note: generation stops at EOS, so realized lengths can be shorter than the
budget. Throughput is always computed from *realized* tokens.
"""
import random

PROMPTS = [
    "The capital of France is",
    "Explain what a CPU does.",
    "The history of artificial intelligence can be traced back to",
    "What is the difference between RAM and storage?",
    "In machine learning, gradient descent is used to",
    "Explain how a transformer model works in detail.",
    "The capital of Japan is",
    "Why is the sky blue?",
    "Describe how a hash table handles collisions.",
    "What causes the seasons on Earth?",
    "Write a short note on the importance of sleep.",
    "How does public key cryptography work?",
    "Summarize the plot of Romeo and Juliet.",
    "What is the role of the operating system kernel?",
    "Explain recursion to a beginner.",
    "Why do airplanes stay in the air?",
]
 
# Fixed filler paragraph (~150-200 tokens) used by the longctx workload.
CONTEXT = (
    "Distributed systems are collections of independent computers that appear "
    "to their users as a single coherent system. They are built to share "
    "resources, to tolerate failures of individual machines, and to scale by "
    "adding more nodes. Designers must deal with partial failure, unreliable "
    "networks, clock drift, and the difficulty of agreeing on a single order "
    "of events. Techniques such as replication, consensus protocols, "
    "partitioning, and caching are combined to balance consistency, "
    "availability, and latency. Observability through metrics, logs, and "
    "traces is essential because failures are often emergent behaviours that "
    "no single component can explain on its own. Operators rely on load "
    "balancers, queues, and back-pressure to keep overload from cascading "
    "through the stack. "
)

MIXED_BUDGETS = [8, 16, 32, 64, 96, 128]
MIXED_WEIGHTS = [3, 3, 3, 2, 1, 1]

def make_workload(name, n, seed):
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        prompt = rng.choice(PROMPTS)
        if name == "uniform":
            budget = 48
        elif name in ("mixed", "longctx"):
            budget = rng.choices(MIXED_BUDGETS, MIXED_WEIGHTS)[0]
        else:
            raise ValueError(f"Unknown workload {name}")
        if name == "longctx":
            prompt = CONTEXT + "Question: " + prompt
        items.append((prompt, budget))

    return items

def poisson_offsets(n, rate, seed):
    """Arrival offsets (seconds from t0) of a Poisson process at `rate` req/s."""
    rng = random.Random(seed + 10_007)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out