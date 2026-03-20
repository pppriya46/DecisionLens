#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter


def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = rank - lower
    return lower_value + (upper_value - lower_value) * weight


def make_request(method, url, body=None, timeout=60):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    status_code = None
    response_body = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response_body = exc.read().decode("utf-8")
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "ok": False,
            "status_code": None,
            "duration_ms": duration_ms,
            "error": str(exc),
            "response": None,
        }

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    parsed = None
    if response_body:
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = response_body

    return {
        "ok": 200 <= status_code < 300,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "error": None,
        "response": parsed,
    }


def summarize_results(endpoint_name, results):
    durations = [item["duration_ms"] for item in results if item["ok"]]
    status_counts = Counter(item["status_code"] for item in results)
    error_count = sum(1 for item in results if not item["ok"])

    summary = {
        "endpoint": endpoint_name,
        "requests": len(results),
        "successful_requests": len(durations),
        "failed_requests": error_count,
        "status_codes": dict(sorted(status_counts.items())),
    }

    if durations:
        summary.update({
            "min_ms": round(min(durations), 2),
            "mean_ms": round(statistics.mean(durations), 2),
            "median_ms": round(statistics.median(durations), 2),
            "p95_ms": round(percentile(durations, 0.95), 2),
            "p99_ms": round(percentile(durations, 0.99), 2),
            "max_ms": round(max(durations), 2),
        })

    first_error = next((item for item in results if not item["ok"]), None)
    if first_error:
        summary["sample_error"] = {
            "status_code": first_error["status_code"],
            "error": first_error["error"],
            "response": first_error["response"],
        }

    return summary


def benchmark_endpoint(base_url, incident_id, endpoint, runs, timeout):
    results = []

    if endpoint == "similar":
        url = f"{base_url}/incidents/{incident_id}/similar"
        for _ in range(runs):
            results.append(make_request("GET", url, timeout=timeout))
        return summarize_results("/incidents/{id}/similar", results)

    if endpoint == "resolve":
        url = f"{base_url}/incidents/{incident_id}/resolve"
        body = {"force_regenerate": True}
        for _ in range(runs):
            results.append(make_request("POST", url, body=body, timeout=timeout))
        return summarize_results("/incidents/{id}/resolve", results)

    raise ValueError(f"Unsupported endpoint: {endpoint}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DecisionLens API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--incident-id", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--endpoint",
        choices=["similar", "resolve", "all"],
        default="all",
        help="Endpoint to benchmark.",
    )
    args = parser.parse_args()

    endpoints = ["similar", "resolve"] if args.endpoint == "all" else [args.endpoint]
    output = {
        "base_url": args.base_url,
        "incident_id": args.incident_id,
        "runs_per_endpoint": args.runs,
        "results": [],
    }

    for endpoint in endpoints:
        summary = benchmark_endpoint(
            base_url=args.base_url.rstrip("/"),
            incident_id=args.incident_id,
            endpoint=endpoint,
            runs=args.runs,
            timeout=args.timeout,
        )
        output["results"].append(summary)

    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
