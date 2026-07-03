from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
JUDGE_DIR = SCRIPT_DIR.parent / "LLM_as_judge"

for path in (SCRIPT_DIR, JUDGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search as faiss_search  # type: ignore
import LLM_as_judge as judge_module  # type: ignore


def slugify(text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not slug:
        return "query"
    return slug[:max_length].strip("-") or "query"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def summarize_labels(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def release_embedder(shared_embedder: Any) -> None:
    del shared_embedder
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_judge_task(
    judge: judge_module.LLMJudge,
    task: Any,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    processed_examples = [
        task.preprocess_example(judge, example) for example in examples
    ]
    prompts = [task.build_prompt(example) for example in processed_examples]
    sampling_params = judge.get_sampling_params(args.model)
    responses = judge.generate(sampling_params, prompts)
    labels = [task.parse_response(response) for response in responses]
    task.print_results(labels, args)
    return processed_examples, responses, labels


def collect_documents(
    descriptor_hits: list[dict[str, Any]],
    labels: list[str],
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    for hit, label in zip(descriptor_hits, labels):
        if label != "yes":
            continue

        for document in hit.get("documents", []):
            doc_id = str(document["doc_id"])
            entry = selected.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "document": document["text"],
                    "matched_descriptors": [],
                    "matched_descriptor_distances": [],
                },
            )
            if hit["descriptor"] not in entry["matched_descriptors"]:
                entry["matched_descriptors"].append(hit["descriptor"])
                entry["matched_descriptor_distances"].append(hit["distance"])

    return sorted(selected.values(), key=lambda row: row["doc_id"])


def build_query_run_dir(root: Path, query: str, index: int) -> Path:
    return root / f"{index:02d}_{slugify(query)}"


def query_artifact_paths(query_dir: Path) -> dict[str, Path]:
    return {
        "search_path": query_dir / "search_results.jsonl",
        "descriptor_path": query_dir / "descriptor_judgements.jsonl",
        "selected_docs_path": query_dir / "selected_documents.jsonl",
        "document_path": query_dir / "document_judgements.jsonl",
        "final_path": query_dir / "final_results.jsonl",
        "summary_path": query_dir / "summary.json",
    }


def run_query_search(
    args: argparse.Namespace,
    query: str,
    index: int,
    shared_index: Any,
    shared_embedder: Any,
) -> dict[str, Any]:
    query_dir = build_query_run_dir(Path(args.output_dir), query, index)
    query_dir.mkdir(parents=True, exist_ok=True)

    paths = query_artifact_paths(query_dir)
    search_path = paths["search_path"]

    if search_path.exists():
        search_rows = load_jsonl(search_path)
        search_payload = (
            search_rows[0] if search_rows else {"query": query, "results": []}
        )
        print(f"Using existing FAISS search results from {search_path}", flush=True)
    else:
        print(f"Running FAISS search...", flush=True)
        search_output = faiss_search.run_search_query(
            shared_index,
            shared_embedder,
            query,
            top_k=args.top_k,
            nprobe=args.nprobe,
            max_distance=args.max_distance,
            max_attempts=args.max_search_attempts,
        )

        if search_output is None:
            search_payload = {"query": query, "results": []}
        else:
            distances, indices = search_output
            search_payload = faiss_search.build_search_result(
                query, distances, indices, shared_index
            )

        write_jsonl(search_path, [search_payload])

    return {
        "query": query,
        "search_payload": search_payload,
        **paths,
    }


def run_query_judgements(
    args: argparse.Namespace,
    query_run: dict[str, Any],
    judge: judge_module.LLMJudge,
) -> dict[str, Any]:
    query = query_run["query"]
    if "search_payload" not in query_run:
        return query_run

    search_payload = query_run["search_payload"]
    search_path = query_run["search_path"]
    descriptor_path = query_run["descriptor_path"]
    selected_docs_path = query_run["selected_docs_path"]
    document_path = query_run["document_path"]
    final_path = query_run["final_path"]
    summary_path = query_run["summary_path"]

    if (
        search_path.exists()
        and descriptor_path.exists()
        and selected_docs_path.exists()
        and document_path.exists()
        and final_path.exists()
        and summary_path.exists()
    ):
        print(f"Using existing pipeline results for query: {query}", flush=True)
        return load_json(summary_path)

    descriptor_rows: list[dict[str, Any]] = []
    selected_documents: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []

    if descriptor_path.exists():
        descriptor_rows = load_jsonl(descriptor_path)
        print(
            f"Using existing descriptor judgements from {descriptor_path}",
            flush=True,
        )

    if selected_docs_path.exists():
        selected_documents = load_jsonl(selected_docs_path)
        print(
            f"Using existing selected documents from {selected_docs_path}",
            flush=True,
        )

    if document_path.exists():
        document_rows = load_jsonl(document_path)
        print(
            f"Using existing document judgements from {document_path}",
            flush=True,
        )
    elif final_path.exists():
        document_rows = load_jsonl(final_path)
        write_jsonl(document_path, document_rows)

    descriptor_labels = [str(row.get("label", "invalid")) for row in descriptor_rows]
    document_labels = [str(row.get("label", "invalid")) for row in document_rows]

    if not search_payload["results"]:
        if not descriptor_path.exists():
            write_jsonl(descriptor_path, [])
        if not selected_docs_path.exists():
            write_jsonl(selected_docs_path, [])
        if not document_path.exists():
            write_jsonl(document_path, [])
        if not final_path.exists():
            write_jsonl(final_path, [])

        summary = {
            "query": query,
            "artifacts": {
                "search_results": str(search_path),
                "descriptor_judgements": str(descriptor_path),
                "selected_documents": str(selected_docs_path),
                "document_judgements": str(document_path),
                "final_results": str(final_path),
            },
            "search": {"results": 0},
            "descriptor_judge": {"labels": {}, "selected_descriptors": 0},
            "documents": {"selected": 0},
            "document_judge": {"labels": {}, "final_documents": 0},
        }
        write_json(summary_path, summary)
        return summary

    if not descriptor_rows:
        print(
            f"Running descriptor judgement for {len(search_payload['results'])} hits...",
            flush=True,
        )
        descriptor_task = judge_module.TASKS["QueryDescriptorMatch"]
        descriptor_examples = [
            {"query": query, "descriptor": hit["descriptor"]}
            for hit in search_payload["results"]
        ]

        _, descriptor_responses, descriptor_labels = run_judge_task(
            judge,
            descriptor_task,
            descriptor_examples,
            argparse.Namespace(query=query, model=args.model),
        )

        descriptor_rows = []
        for hit, response, label in zip(
            search_payload["results"], descriptor_responses, descriptor_labels
        ):
            descriptor_rows.append(
                {
                    "query": query,
                    "descriptor": hit["descriptor"],
                    "distance": hit["distance"],
                    "documents": hit["documents"],
                    "response": response,
                    "label": label,
                }
            )
        write_jsonl(descriptor_path, descriptor_rows)
        descriptor_labels = [
            str(row.get("label", "invalid")) for row in descriptor_rows
        ]

    if not selected_documents:
        selected_documents = collect_documents(
            search_payload["results"], descriptor_labels
        )
        write_jsonl(selected_docs_path, selected_documents)

    document_task = judge_module.TASKS["QueryDocMatch"]
    document_examples = [
        {"query": query, "document": row["document"]} for row in selected_documents
    ]

    if not document_rows and document_examples:
        _, document_responses, document_labels = run_judge_task(
            judge,
            document_task,
            document_examples,
            argparse.Namespace(query=query, model=args.model),
        )
        print(
            f"Running document judgement for {len(selected_documents)} hits...",
            flush=True,
        )
        document_rows = []
        for row, response, label in zip(
            selected_documents, document_responses, document_labels
        ):
            document_rows.append(
                {
                    "query": query,
                    **row,
                    "response": response,
                    "label": label,
                }
            )
        write_jsonl(document_path, document_rows)
        write_jsonl(final_path, document_rows)
        document_labels = [str(row.get("label", "invalid")) for row in document_rows]
    elif document_rows and not final_path.exists():
        write_jsonl(final_path, document_rows)
    elif document_rows and not document_path.exists():
        write_jsonl(document_path, document_rows)

    summary = {
        "query": query,
        "artifacts": {
            "search_results": str(search_path),
            "descriptor_judgements": str(descriptor_path),
            "selected_documents": str(selected_docs_path),
            "document_judgements": str(document_path),
            "final_results": str(final_path),
        },
        "search": {"results": len(search_payload["results"])},
        "descriptor_judge": {
            "labels": summarize_labels(descriptor_labels),
            "selected_descriptors": sum(
                1 for label in descriptor_labels if label == "yes"
            ),
        },
        "documents": {"selected": len(selected_documents)},
        "document_judge": {
            "labels": summarize_labels(document_labels),
            "final_documents": len(document_rows),
        },
    }
    write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FAISS search -> descriptor judge -> document judge pipeline."
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to the document JSONL."
    )
    parser.add_argument(
        "--query", required=True, help="Query string. Use | for multiple queries."
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline_results",
        help="Directory where all pipeline artifacts will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory for caching the embedding and judge model files.",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Path to the FAISS index. Defaults to <output-dir>/index.faiss.",
    )
    parser.add_argument(
        "--embeddings-path",
        default=None,
        help="Path to the cached embeddings. Defaults to <output-dir>/embeddings.npy.",
    )

    parser.add_argument(
        "--build-index", action="store_true", help="Build or rebuild the index."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the FAISS index even if it already exists.",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Force recomputing descriptor embeddings.",
    )
    parser.add_argument(
        "--descriptor-type",
        choices=["raw", "harmonized"],
        default="raw",
        help="Descriptor field to index from the input documents.",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Optional cap for input docs."
    )

    parser.add_argument(
        "--index-type", default="IndexIVFFlat", help="FAISS index type."
    )
    parser.add_argument("--nlist", type=int, default=100, help="FAISS IVF list count.")
    parser.add_argument(
        "--dimension", type=int, default=1024, help="Embedding dimension."
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Descriptor search fan-out."
    )
    parser.add_argument("--nprobe", type=int, default=10, help="FAISS IVF probe count.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Optional distance threshold for search hits.",
    )
    parser.add_argument(
        "--max-search-attempts",
        type=int,
        default=10,
        help="Maximum retry count when expanding top-k under a distance threshold.",
    )

    parser.add_argument(
        "--backend",
        choices=["local", "api"],
        default="local",
        help="Judge backend.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Next-80B-A3B-Instruct",
        help="Judge model name or path.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=128000,
        help="Judge model context length.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum judge output tokens.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force vLLM eager mode. Defaults to on for ROCm and off otherwise.",
    )
    parser.add_argument(
        "--api-base-url", default=None, help="Optional OpenAI-compatible API URL."
    )
    parser.add_argument(
        "--api-key", default=None, help="Optional OpenAI-compatible API key."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    if args.index_path is None:
        args.index_path = str(output_dir / "index.faiss")
    if args.embeddings_path is None:
        args.embeddings_path = str(output_dir / "embeddings.npy")

    queries = [query.strip() for query in args.query.split("|") if query.strip()]
    if not queries:
        raise ValueError("Provide at least one non-empty query.")

    print("Building or loading the FAISS index...", flush=True)
    shared_index, shared_embedder = faiss_search.build_or_load_index(args)
    print("FAISS index ready.", flush=True)

    query_runs = []
    for index, query in enumerate(queries, start=1):
        print(f"\n=== Query {index}/{len(queries)}: {query} ===", flush=True)
        query_dir = build_query_run_dir(output_dir, query, index)
        paths = query_artifact_paths(query_dir)
        if all(path.exists() for path in paths.values()):
            print(
                f"Skipping query '{query}' because all result files already exist.",
                flush=True,
            )
            query_runs.append(load_json(paths["summary_path"]))
            continue
        query_runs.append(
            run_query_search(args, query, index, shared_index, shared_embedder)
        )

    release_embedder(shared_embedder)
    shared_embedder = None
    print("Initializing the LLM judge...", flush=True)
    judge = judge_module.LLMJudge(args)

    run_summaries = []
    for query_run in query_runs:
        run_summaries.append(run_query_judgements(args, query_run, judge))

    write_json(output_dir / "pipeline_summary.json", {"runs": run_summaries})

    print("\nPipeline complete.", flush=True)
    for run in run_summaries:
        print(
            f"- {run['query']}: {run['descriptor_judge']['selected_descriptors']} descriptors kept, "
            f"{run['document_judge']['final_documents']} documents judged",
            flush=True,
        )


if __name__ == "__main__":
    main()
