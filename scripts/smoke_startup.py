"""Verifica inicialização HTTP local; não publica nem simula sessão de usuário.

As páginas e seus cálculos são exercitados separadamente pelos AppTests.
Somente biblioteca padrão; funciona no Windows e no Linux.
"""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


def smoke_startup(root: Path, timeout: float = 40) -> None:
    root = root.resolve()
    entrypoint = root / "streamlit_app.py"
    if not entrypoint.is_file():
        raise ValueError("streamlit_app.py não encontrado na raiz informada.")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = os.environ.copy()
    environment["ENERGY_FORECAST_ALLOW_DEMO"] = "false"
    command = [
        sys.executable, "-m", "streamlit", "run", str(entrypoint),
        "--server.headless=true", "--server.address=127.0.0.1",
        f"--server.port={port}", "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + timeout
            health_url = f"http://127.0.0.1:{port}/_stcore/health"
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError("Servidor encerrou na inicialização:\n" + log.read().decode("utf-8", errors="replace")[-4000:])
                try:
                    with urlopen(health_url, timeout=2) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            break
                except (URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Servidor não ficou saudável no prazo de inicialização.")
                time.sleep(0.2)
            for route in ("/", "/desempenho", "/sazonalidade"):
                with urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as response:
                    if response.status != 200 or b"<html" not in response.read().lower():
                        raise RuntimeError(f"Resposta HTTP inesperada em {route}.")
            print("Inicialização HTTP: saúde ok; três rotas responderam 200. Demonstração desativada.")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    smoke_startup(args.root)


if __name__ == "__main__":
    main()
