from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def npm_executable() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def install_dependencies() -> None:
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], PROJECT_ROOT)
    run_command([npm_executable(), "install"], FRONTEND_DIR)


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the EduAgentX backend and frontend together.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for both development servers.")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port.")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port.")
    parser.add_argument("--install-deps", action="store_true", help="Install backend and frontend dependencies first.")
    parser.add_argument("--skip-backend", action="store_true", help="Start only the frontend dev server.")
    parser.add_argument("--skip-frontend", action="store_true", help="Start only the backend API server.")
    args = parser.parse_args()

    if args.skip_backend and args.skip_frontend:
        parser.error("Cannot skip both backend and frontend.")

    if args.install_deps:
        install_dependencies()

    base_env = os.environ.copy()
    base_env.setdefault("ENABLE_MOCK_LLM", "true")
    base_env.setdefault("ENABLE_SPARK_FALLBACK", "true")

    processes: list[subprocess.Popen[str]] = []
    try:
        if not args.skip_backend:
            backend_command = [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app:app",
                "--host",
                args.host,
                "--port",
                str(args.backend_port),
            ]
            processes.append(subprocess.Popen(backend_command, cwd=PROJECT_ROOT, env=base_env))

        if not args.skip_frontend:
            frontend_env = base_env.copy()
            frontend_env["NEXT_PUBLIC_API_BASE_URL"] = f"http://{args.host}:{args.backend_port}"
            frontend_command = [
                npm_executable(),
                "run",
                "dev",
                "--",
                "--hostname",
                args.host,
                "--port",
                str(args.frontend_port),
            ]
            processes.append(subprocess.Popen(frontend_command, cwd=FRONTEND_DIR, env=frontend_env))

        print(f"Backend:  http://{args.host}:{args.backend_port}" if not args.skip_backend else "Backend:  skipped")
        print(f"Frontend: http://{args.host}:{args.frontend_port}" if not args.skip_frontend else "Frontend: skipped")
        print("Press Ctrl+C to stop both services.")

        while True:
            for process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    return exit_code
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            terminate_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
