from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jawbreaker.analyzers import (  # noqa: E402
    analysis_to_prediction,
    build_llama_cpp_analyzer,
    build_transformers_analyzer,
    has_unsafe_action,
    heuristic_analyzer,
    load_prediction_jsonl,
    prediction_file_analyzer,
    prediction_to_analysis,
    repair_prediction,
    should_apply_heuristic_guard,
    validate_prediction,
    write_predictions,
)
from jawbreaker.schema import ScamAnalysis  # noqa: E402
from jawbreaker.schema import RISK_LEVELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Jawbreaker scam-risk evals.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("scam_eval.jsonl"))
    parser.add_argument(
        "--backend",
        choices=["heuristic", "predictions", "llama-cpp", "transformers"],
        default="heuristic",
    )
    parser.add_argument("--predictions", type=Path, help="JSONL predictions for --backend predictions.")
    parser.add_argument("--predictions-out", type=Path, help="Write predictions as JSONL.")
    parser.add_argument("--json-out", type=Path, help="Write metrics as JSON.")
    parser.add_argument("--limit", type=int, help="Limit number of eval cases for smoke tests.")
    parser.add_argument("--show-failures", type=int, default=5, help="Failures to print per category.")
    parser.add_argument(
        "--apply-safety-guard",
        action="store_true",
        help="Apply the same deterministic undercall guard used by the app before scoring.",
    )

    parser.add_argument("--model-path", type=Path, help="GGUF path for --backend llama-cpp.")
    parser.add_argument("--chat-format", help="Optional llama-cpp-python chat_format.")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--offload-kqv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--op-offload", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model-id", default="openbmb/MiniCPM4.1-8B", help="HF model id for --backend transformers.")
    parser.add_argument("--adapter-id", help="Optional PEFT adapter id for --backend transformers.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="auto", help="Transformers dtype.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="eager", help="Transformers attention implementation.")
    return parser.parse_args()


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    ids = set()
    errors = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc}")
            continue

        missing = {"id", "category", "input", "expected_risk_level", "expected_scam_type", "expected_tactics"} - set(row)
        if missing:
            errors.append(f"line {line_number}: missing fields: {sorted(missing)}")

        if row.get("id") in ids:
            errors.append(f"line {line_number}: duplicate id: {row.get('id')}")
        ids.add(row.get("id"))

        if row.get("expected_risk_level") not in RISK_LEVELS:
            errors.append(f"line {line_number}: invalid expected_risk_level: {row.get('expected_risk_level')}")

        if not isinstance(row.get("expected_tactics"), list):
            errors.append(f"line {line_number}: expected_tactics must be a list")

        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break

    if errors:
        raise SystemExit("Eval dataset validation failed:\n" + "\n".join(errors))
    return rows


def tactic_recall(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 1.0
    expected_set = set(expected)
    actual_set = set(actual)
    return len(expected_set & actual_set) / len(expected_set)


def build_analyzer(args: argparse.Namespace):
    if args.backend == "heuristic":
        return lambda row: heuristic_analyzer(row["input"])

    if args.backend == "predictions":
        if not args.predictions:
            raise SystemExit("--predictions is required with --backend predictions")
        predictions = load_prediction_jsonl(args.predictions)
        return lambda row: prediction_file_analyzer(predictions, row["id"])

    if args.backend == "llama-cpp":
        if not args.model_path:
            raise SystemExit("--model-path is required with --backend llama-cpp")
        analyzer = build_llama_cpp_analyzer(
            args.model_path,
            chat_format=args.chat_format,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
            offload_kqv=args.offload_kqv,
            op_offload=args.op_offload,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return lambda row: analyzer(row["input"])

    if args.backend == "transformers":
        analyzer = build_transformers_analyzer(
            args.model_id,
            adapter_id=args.adapter_id,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
        return lambda row: analyzer(row["input"])

    raise SystemExit(f"Unsupported backend: {args.backend}")


def apply_safety_guard(message: str, prediction: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    model_analysis = prediction_to_analysis(prediction)
    heuristic = ScamAnalysis.from_heuristics(message)
    if should_apply_heuristic_guard(message, model_analysis, heuristic, validation_errors=[]):
        return analysis_to_prediction(heuristic), True

    return prediction, False


def score_rows(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], elapsed: float) -> dict[str, Any]:
    risk_correct = 0
    scam_type_correct = 0
    dangerous_as_safe = []
    dangerous_as_needs_check = []
    suspicious_as_safe = []
    safe_as_dangerous = []
    unsafe_actions = []
    invalid_predictions = []
    model_errors = []
    safety_guard_promotions = []
    recalls = []
    risk_confusion: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()
    failures_by_category: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        case_id = row["id"]
        prediction = predictions[case_id]
        if "_jawbreaker_model_error" in prediction:
            model_errors.append({"id": case_id, "error": str(prediction["_jawbreaker_model_error"])})
        if "_jawbreaker_safety_guard" in prediction:
            safety_guard_promotions.append({"id": case_id, "from": str(prediction["_jawbreaker_safety_guard"])})
        validation_errors = validate_prediction(prediction)
        if validation_errors:
            invalid_predictions.append({"id": case_id, "errors": validation_errors})

        actual_risk = prediction.get("risk_level", "invalid")
        actual_scam_type = prediction.get("scam_type", "invalid")
        actual_tactics = prediction.get("tactics", [])
        if not isinstance(actual_tactics, list):
            actual_tactics = []

        expected_risk = row["expected_risk_level"]
        category = row["category"]
        risk_ok = actual_risk == expected_risk
        type_ok = actual_scam_type == row["expected_scam_type"]
        recall = tactic_recall(row["expected_tactics"], [str(tactic) for tactic in actual_tactics])

        risk_correct += int(risk_ok)
        scam_type_correct += int(type_ok)
        recalls.append(recall)
        risk_confusion[(expected_risk, str(actual_risk))] += 1
        category_counts[category] += 1
        category_correct[category] += int(risk_ok)

        if expected_risk == "dangerous" and actual_risk == "safe":
            dangerous_as_safe.append(case_id)
        if expected_risk == "dangerous" and actual_risk == "needs_check":
            dangerous_as_needs_check.append(case_id)
        if expected_risk == "suspicious" and actual_risk == "safe":
            suspicious_as_safe.append(case_id)
        if expected_risk == "safe" and actual_risk in {"dangerous", "suspicious"}:
            safe_as_dangerous.append(case_id)
        if has_unsafe_action(str(prediction.get("safest_action", ""))):
            unsafe_actions.append(case_id)
        if not risk_ok:
            failures_by_category[category].append(f"{case_id} expected={expected_risk} actual={actual_risk}")

    total = len(rows)
    return {
        "cases": total,
        "risk_level_correct": risk_correct,
        "risk_level_accuracy": risk_correct / total,
        "scam_type_correct": scam_type_correct,
        "scam_type_accuracy": scam_type_correct / total,
        "mean_tactic_recall": sum(recalls) / len(recalls),
        "dangerous_as_safe": dangerous_as_safe,
        "dangerous_as_needs_check": dangerous_as_needs_check,
        "suspicious_as_safe": suspicious_as_safe,
        "safe_as_dangerous_or_suspicious": safe_as_dangerous,
        "unsafe_action_violations": unsafe_actions,
        "invalid_predictions": invalid_predictions,
        "model_errors": model_errors,
        "safety_guard_promotions": safety_guard_promotions,
        "elapsed_seconds": elapsed,
        "risk_confusion": {f"{expected}->{actual}": count for (expected, actual), count in sorted(risk_confusion.items())},
        "category_risk_accuracy": {
            category: {
                "correct": category_correct[category],
                "total": count,
                "accuracy": category_correct[category] / count,
            }
            for category, count in sorted(category_counts.items())
        },
        "failures_by_category": {category: failures for category, failures in sorted(failures_by_category.items())},
    }


def print_report(metrics: dict[str, Any], show_failures: int) -> None:
    total = metrics["cases"]
    print(f"cases={total}")
    print(
        "risk_level_accuracy="
        f"{metrics['risk_level_correct']}/{total} ({metrics['risk_level_accuracy']:.1%})"
    )
    print(
        "scam_type_accuracy="
        f"{metrics['scam_type_correct']}/{total} ({metrics['scam_type_accuracy']:.1%})"
    )
    print(f"mean_tactic_recall={metrics['mean_tactic_recall']:.1%}")
    print(f"dangerous_as_safe={len(metrics['dangerous_as_safe'])} {metrics['dangerous_as_safe']}")
    print(
        "dangerous_as_needs_check="
        f"{len(metrics['dangerous_as_needs_check'])} {metrics['dangerous_as_needs_check']}"
    )
    print(f"suspicious_as_safe={len(metrics['suspicious_as_safe'])} {metrics['suspicious_as_safe']}")
    print(
        "safe_as_dangerous_or_suspicious="
        f"{len(metrics['safe_as_dangerous_or_suspicious'])} {metrics['safe_as_dangerous_or_suspicious']}"
    )
    print(f"unsafe_action_violations={len(metrics['unsafe_action_violations'])} {metrics['unsafe_action_violations']}")
    print(f"invalid_predictions={len(metrics['invalid_predictions'])} {metrics['invalid_predictions'][:show_failures]}")
    print(f"model_errors={len(metrics['model_errors'])} {metrics['model_errors'][:show_failures]}")
    print(
        "safety_guard_promotions="
        f"{len(metrics['safety_guard_promotions'])} {metrics['safety_guard_promotions'][:show_failures]}"
    )
    print(f"elapsed_seconds={metrics['elapsed_seconds']:.3f}")

    print("\nrisk_confusion expected->actual:")
    for pair, count in metrics["risk_confusion"].items():
        expected, actual = pair.split("->", 1)
        print(f"  {expected:12s} -> {actual:12s} {count}")

    print("\ncategory_risk_accuracy:")
    for category, result in metrics["category_risk_accuracy"].items():
        print(
            f"  {category:24s} {result['correct']:2d}/{result['total']:2d} "
            f"({result['accuracy']:.1%})"
        )

    if metrics["failures_by_category"]:
        print("\nfirst_failures_by_category:")
        for category, failures in metrics["failures_by_category"].items():
            print(f"  {category}:")
            for failure in failures[:show_failures]:
                print(f"    {failure}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset, args.limit)
    analyzer = build_analyzer(args)
    predictions = {}

    started = perf_counter()
    for index, row in enumerate(rows, start=1):
        print(f"eval case {index}/{len(rows)} id={row['id']}", flush=True)
        try:
            prediction = repair_prediction(analyzer(row))
        except Exception as exc:
            prediction = heuristic_analyzer(row["input"])
            prediction["_jawbreaker_model_error"] = repr(exc)
            prediction = repair_prediction(prediction)
        if args.apply_safety_guard:
            guarded_prediction, promoted = apply_safety_guard(row["input"], prediction)
            if promoted:
                guarded_prediction["_jawbreaker_safety_guard"] = prediction.get("risk_level", "unknown")
            prediction = repair_prediction(guarded_prediction)
        predictions[row["id"]] = prediction
    elapsed = perf_counter() - started

    metrics = score_rows(rows, predictions, elapsed)
    print_report(metrics, args.show_failures)

    if args.predictions_out:
        write_predictions(args.predictions_out, rows, predictions)
    if args.json_out:
        args.json_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
