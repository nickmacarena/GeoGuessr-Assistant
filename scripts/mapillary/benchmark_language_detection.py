#!/usr/bin/env python3
"""
Benchmark fastText language detection latency.

Measures inference speed for single predictions and batch processing.
"""

import time
from language_detector import LanguageDetector
import statistics


def benchmark_single_inference(detector, texts, iterations=100):
    """Benchmark single text predictions."""
    print("\n" + "="*60)
    print("SINGLE INFERENCE BENCHMARK")
    print("="*60)

    latencies = []

    for text in texts[:5]:  # Use first 5 texts
        text_latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            result = detector.detect(text)
            end = time.perf_counter()
            text_latencies.append((end - start) * 1000)  # Convert to ms

        avg_latency = statistics.mean(text_latencies)
        p50 = statistics.median(text_latencies)
        p95 = statistics.quantiles(text_latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(text_latencies, n=100)[98]  # 99th percentile

        latencies.extend(text_latencies)

        print(f"\nText: '{text[:40]}'")
        print(f"  Avg:  {avg_latency:.2f} ms")
        print(f"  p50:  {p50:.2f} ms")
        print(f"  p95:  {p95:.2f} ms")
        print(f"  p99:  {p99:.2f} ms")

    # Overall stats
    overall_avg = statistics.mean(latencies)
    overall_p50 = statistics.median(latencies)
    overall_p95 = statistics.quantiles(latencies, n=20)[18]
    overall_p99 = statistics.quantiles(latencies, n=100)[98]

    print(f"\n{'Overall Performance:'}")
    print(f"  Average:  {overall_avg:.2f} ms")
    print(f"  p50:      {overall_p50:.2f} ms")
    print(f"  p95:      {overall_p95:.2f} ms")
    print(f"  p99:      {overall_p99:.2f} ms")

    return overall_avg


def benchmark_batch_processing(detector, texts, batch_sizes=[10, 50, 100]):
    """Benchmark batch processing."""
    print("\n" + "="*60)
    print("BATCH PROCESSING BENCHMARK")
    print("="*60)

    for batch_size in batch_sizes:
        batch = texts[:batch_size]

        start = time.perf_counter()
        results = detector.detect_batch(batch)
        end = time.perf_counter()

        total_time = (end - start) * 1000  # ms
        per_item = total_time / batch_size

        print(f"\nBatch size: {batch_size}")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Per item:   {per_item:.2f} ms")
        print(f"  Throughput: {1000/per_item:.0f} items/sec")


def benchmark_model_loading():
    """Benchmark model loading time."""
    print("\n" + "="*60)
    print("MODEL LOADING BENCHMARK")
    print("="*60)

    iterations = 5
    loading_times = []

    for i in range(iterations):
        start = time.perf_counter()
        detector = LanguageDetector()
        end = time.perf_counter()
        loading_time = (end - start) * 1000
        loading_times.append(loading_time)
        print(f"  Load #{i+1}: {loading_time:.2f} ms")

    avg_loading = statistics.mean(loading_times)
    print(f"\nAverage loading time: {avg_loading:.2f} ms")

    return avg_loading


def main():
    print("fastText Language Detection Latency Benchmark")
    print("="*60)

    # Test texts from real traffic signs
    test_texts = [
        "STOP",
        "ONE WAY",
        "禁止停车",
        "REDUCE SPEED NOW",
        "Curva peligrosa Disminuya su Velocidad",
        "FAIRDALE AVE",
        "35",
        "止まれ",
        "ARRÊT",
        "ATENCION 1000mts SALOAGANONES",
        "Bruxelles par N4",
        "Neuburg 72 Zuchering",
        "鲁班路 北 558 Luban Rd. N",
        "パールホテル 3",
        "ZONA MILITAR",
        "NO GIRAR ALA IZQUIERDA",
        "VEHICULOS PESADOS",
        "2 TONS",
        "Lieh och",
        "ONEWAY"
    ]

    # Benchmark model loading
    avg_loading = benchmark_model_loading()

    # Initialize detector (keep it loaded for inference benchmarks)
    print("\nInitializing detector for inference benchmarks...")
    detector = LanguageDetector()

    # Benchmark single inference
    avg_latency = benchmark_single_inference(detector, test_texts, iterations=100)

    # Benchmark batch processing
    benchmark_batch_processing(detector, test_texts * 10, batch_sizes=[10, 50, 100, 500])

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model loading:      {avg_loading:.2f} ms (one-time cost)")
    print(f"Avg inference:      {avg_latency:.2f} ms per prediction")
    print(f"Expected throughput: {1000/avg_latency:.0f} predictions/sec")
    print("\nRecommendation:")
    print("  • Load model once at application startup")
    print("  • Reuse detector instance across requests")
    print(f"  • For real-time use: latency ~{avg_latency:.1f}ms is suitable for most applications")
    print("="*60)


if __name__ == "__main__":
    main()