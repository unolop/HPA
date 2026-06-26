from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    path = Path(__file__).resolve().with_name("human_analysis.ipynb")
    with path.open() as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb, timeout=1200, kernel_name="python3")
    client.execute(cwd=str(path.parent))
    with path.open("w") as f:
        nbformat.write(nb, f)
    print(f"executed {path}")


if __name__ == "__main__":
    main()
