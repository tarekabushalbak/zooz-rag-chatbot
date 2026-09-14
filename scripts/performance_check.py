import statistics
import time

try:
    from scripts.rag_engine import ask_zooz
except ImportError:
    from rag_engine import ask_zooz

MODEL_QUESTIONS = [
    "מה השירותים של ZOOZ?",
    "מי זה ארי מנור?",
    "עם אילו לקוחות ZOOZ עבדה?",
    "מהי חדשנות שיטתית ב-ZOOZ?",
]

LOCAL_QUESTIONS = [
    "איך יוצרים קשר עם ZOOZ?",
]


def run_question(question):
    started = time.perf_counter()
    answer, sources = ask_zooz(question)
    elapsed = time.perf_counter() - started
    technical_error = "אירעה שגיאה טכנית" in (answer or "")
    print(f"\nQUESTION: {question}")
    print(f"TIME: {elapsed:.2f}s")
    print(f"STATUS: {'ERROR' if technical_error else 'OK'}")
    print(f"ANSWER: {answer}")
    print("SOURCES:")
    for source in sources:
        print(f"  - {source}")
    return elapsed, technical_error


def main():
    print("ZOOZ RAG performance check")
    print("The first model-backed question includes cold-start/model-loading overhead.\n")

    model_times = []
    errors = 0
    for question in MODEL_QUESTIONS:
        elapsed, technical_error = run_question(question)
        model_times.append(elapsed)
        errors += int(technical_error)

    local_times = []
    for question in LOCAL_QUESTIONS:
        elapsed, technical_error = run_question(question)
        local_times.append(elapsed)
        errors += int(technical_error)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"Cold model request: {model_times[0]:.2f}s")
    if len(model_times) > 1:
        warm = model_times[1:]
        print(f"Warm model median: {statistics.median(warm):.2f}s")
        print(f"Warm model average: {statistics.mean(warm):.2f}s")
        print(f"Warm model max: {max(warm):.2f}s")
    if local_times:
        print(f"Local deterministic average: {statistics.mean(local_times):.2f}s")
    print(f"Technical errors: {errors}")
    print("=" * 72)


if __name__ == "__main__":
    main()
