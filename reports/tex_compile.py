import subprocess
import sys
from pathlib import Path


def compile_tex_to_pdf(tex_path: str) -> Path:
    tex_path = Path(tex_path).resolve()

    if not tex_path.exists():
        raise FileNotFoundError(f"No such file: {tex_path}")
    if tex_path.suffix.lower() != ".tex":
        raise ValueError(f"Expected a .tex file, got: {tex_path.name}")

    output_dir = tex_path.parent / tex_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",   
        "-halt-on-error",             
        f"-output-directory={output_dir}",
        str(tex_path),
    ]

    result = None
    for _ in range(2):
        result = subprocess.run(
            cmd,
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
        )

    pdf_path = output_dir / (tex_path.stem + ".pdf")

    if result.returncode != 0 or not pdf_path.exists():
        print("---- pdflatex output (tail) ----")
        print(result.stdout[-3000:])
        log_path = output_dir / (tex_path.stem + ".log")
        raise RuntimeError(f"LaTeX compilation failed. See log: {log_path}")

    return pdf_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: py r.py <file.tex>")
        sys.exit(1)

    try:
        pdf = compile_tex_to_pdf(sys.argv[1])
        print(f"PDF created at: {pdf}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)