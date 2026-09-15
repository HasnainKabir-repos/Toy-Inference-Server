"""
Minimal Flask server around the toy ContinuousBatchEngine, for load-testing
with `hey` the same way you load-tested vLLM's CPU backend in Week 1.
 
Apply run_py_patch.md to run.py first (thread-safe RequestQueue +
Request.done_event), then run this alongside it:
 
    python server.py
 
Then in another terminal:
 
    hey -z 30s -c 10 -m POST \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Explain what a CPU does.", "max_new_tokens": 30}' \
        http://localhost:8000/generate
"""
import threading
import time
from flask import Flask, request as flask_request, jsonify

from run import (
    Request,
    ContinuousBatchEngine,
    model,
    tokenizer,
    MAX_BATCH_SIZE
)

app = Flask(__name__)
engine = ContinuousBatchEngine(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=MAX_BATCH_SIZE
)

_next_id_lock = threading.Lock()
_next_id = 1

def _serve_forever(engine, poll_interval=0.001):
    """
    Background thread: runs the same admit -> decode -> finish cycle as
    engine.run(), but never stops when the queue and batch drain to empty --
    it idles briefly and keeps polling for new requests, since this is now
    a long-lived server process rather than a one-shot batch job.
    """
    while True:
        if engine.waiting_queue.empty() and engine.active_batch.empty:
            time.sleep(poll_interval)
            continue
        engine.admit_requests()
        engine.decode_active_batch()
        engine.finish_requests()

@app.route("/generate", methods=["POST"])
def generate():
    global _next_id

    body = flask_request.get_json(force=True)
    prompt = body["prompt"]
    max_new_tokens = body.get("max_new_tokens", 50)

    with _next_id_lock:
        request_id = _next_id
        _next_id += 1

    req = Request(
        request_id = request_id,
        prompt = prompt,
        max_new_tokens=max_new_tokens
    )

    arrival = time.perf_counter()
    engine.add_request(req)

    req.done_event.wait()

    return jsonify({
        "request_id": req.request_id,
        "text": req.text,
        "generated_tokens": req.generated_token_count,
        "finish_reason": req.finish_reason,
        "waiting_time": req.waiting_time,
        "ttft": req.ttft,
        "decode_time": req.decode_time,
        "total_time": req.total_time,
        "server_observed_latency": time.perf_counter() - arrival
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Cheap analog to vLLM's Prometheus gauges (num_requests_waiting,
    num_requests_running) -- poll this from a separate script while `hey`
    is running to plot queue depth over time, same as you did with
    Grafana in Week 1.
    """
    return jsonify({
        "queue_depth": len(engine.waiting_queue),
        "active_batch_size": len(engine.active_batch),
        "completed": len(engine.completed_requests),
    })

if __name__ == "__main__":
    engine_thread = threading.Thread(
        target=_serve_forever,
        args=(engine,),
        daemon=True
    )
    engine_thread.start()

    app.run(host="0.0.0.0", port=8000, threaded=True)