from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def npm_executable() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def run_step(label: str, command: list[str], cwd: Path) -> None:
    print(f"\n==> {label}")
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EduAgentX automated backend and frontend checks.")
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend pytest checks.")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend production build checks.")
    parser.add_argument("--install-deps", action="store_true", help="Install backend and frontend dependencies first.")
    args = parser.parse_args()

    if args.skip_backend and args.skip_frontend:
        parser.error("Cannot skip both backend and frontend.")

    try:
        if args.install_deps:
            run_step("Install backend dependencies", [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], PROJECT_ROOT)
            run_step("Install frontend dependencies", [npm_executable(), "install"], FRONTEND_DIR)

        if not args.skip_backend:
            run_step("Run backend tests", [sys.executable, "-m", "pytest", "backend/tests"], PROJECT_ROOT)

        if not args.skip_frontend:
            run_step("Build frontend", [npm_executable(), "run", "check"], FRONTEND_DIR)
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
