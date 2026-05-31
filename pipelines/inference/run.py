"""
Entry point for the inference service.

Usage:
    python -m pipelines.inference.run
    python -m pipelines.inference.run --host 0.0.0.0 --port 8080
    uvicorn pipelines.inference.app:app --reload
"""
import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the JobAnalysis inference API.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    uvicorn.run(
        "pipelines.inference.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
