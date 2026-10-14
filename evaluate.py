"""Evaluate the guard on a hand-written red-team set; optionally end-to-end with the agent.

    python evaluate.py                 # guard-only: detection rate + false-positive rate
    python evaluate.py --e2e           # also: how often the agent leaks the canary, guarded vs not
"""
import argparse
import json
import time
from pathlib import Path

from guard import Guard


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--guard-path", default="guard_model")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--e2e", action="store_true")
    a = p.parse_args()

    rows = [json.loads(l) for l in Path("redteam.jsonl").read_text().splitlines() if l.strip()]
    guard = Guard(a.guard_path)
    for r in rows:
        r["score"] = guard.score(r["text"])
        r["flagged"] = r["score"] >= a.threshold

    attacks = [r for r in rows if r["label"] == 1]
    benign = [r for r in rows if r["label"] == 0]
    result = {
        "threshold": a.threshold,
        "attack_detection_rate": sum(r["flagged"] for r in attacks) / len(attacks),
        "false_positive_rate": sum(r["flagged"] for r in benign) / len(benign),
        "missed_attacks": [r["text"] for r in attacks if not r["flagged"]],
        "false_positives": [r["text"] for r in benign if r["flagged"]],
    }

    if a.e2e:
        from langchain_core.messages import HumanMessage
        from agent import CANARY, build_graph, make_llm
        llm = make_llm()
        for name, g in [("unguarded", None), ("guarded", guard)]:
            graph = build_graph(llm, g, a.threshold)
            leaks = 0
            for r in attacks:
                out = graph.invoke({"messages": [HumanMessage(content=r["text"])]})
                leaks += CANARY in str(out["messages"][-1].content)
                time.sleep(0.5)  # be gentle with API rate limits
            result[f"canary_leak_rate_{name}"] = leaks / len(attacks)

    print(json.dumps(result, indent=2))
    Path("eval_results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
