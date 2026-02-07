import argparse
from typing import Any

from answer import answer_question


def _safe_source(metadata: dict[str, Any]) -> str:
    # LangChain DirectoryLoader typically stores file path in `source`
    return str(metadata.get("source", "(unknown source)"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question against the local Chroma RAG index (vector_db/)."
    )
    parser.add_argument("question", help="Question to ask")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the retrieved document sources after the answer",
    )
    args = parser.parse_args()

    answer, docs = answer_question(args.question, history=[])
    print(answer)

    if args.show_sources:
        print("\n---\nSources:")
        for i, doc in enumerate(docs, start=1):
            print(f"{i}. {_safe_source(doc.metadata)}")


if __name__ == "__main__":
    main()
