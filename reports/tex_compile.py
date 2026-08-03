import argparse
import subprocess
import sys
from pathlib import Path


def compile_tex(
    tex_file: str,
    engine: str = "latexmk",
    bibtex: bool = False,
    output_dir: str | None = None,
    clean: bool = False,
) -> int:
    """
    Compile a .tex file into a PDF.

    Parameters
    ----------
    tex_file : str
        Path to the .tex file.
    engine : str
        LaTeX engine to use ('pdflatex', 'xelatex', 'lualatex', or 'latexmk').
    bibtex : bool
        Whether to run BibTeX (only relevant when engine is not 'latexmk').
    output_dir : str, optional
        Directory for auxiliary/output files. If using latexmk, this will be
        passed as -outdir.
    clean : bool
        Remove auxiliary files after compilation (latexmk -c or engine specific).

    Returns
    -------
    int
        Return code of the compilation process.
    """
    tex_path = Path(tex_file).resolve()
    if not tex_path.exists():
        print(f"Error: File not found – {tex_path}", file=sys.stderr)
        return 1

    if tex_path.suffix != ".tex":
        print("Warning: Input file does not have a .tex extension.", file=sys.stderr)

    if engine == "latexmk":
        cmd = ["latexmk", "-pdf", f"-pdflatex={engine}"]
        cmd = ["latexmk", "-pdf"]
        if output_dir:
            cmd.extend(["-outdir", output_dir])
        if clean:
            cmd.append("-c")  
        cmd.append(str(tex_path))
    else:
        cmd = [engine, "-interaction=nonstopmode", "-halt-on-error"]
        if output_dir:
            cmd.extend(["-output-directory", output_dir])
        cmd.append(str(tex_path))

        if bibtex and not clean:
            bib_cmd = ["bibtex", tex_path.stem]
            pass  

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("Compilation failed.", file=sys.stderr)
        else:
            print("Compilation successful.")
        return result.returncode
    except FileNotFoundError:
        print(
            f"Error: '{engine}' not found. Is a LaTeX distribution installed and in PATH?",
            file=sys.stderr,
        )
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Compile a LaTeX .tex file to PDF."
    )
    parser.add_argument("texfile", help="Path to the .tex file")
    parser.add_argument(
        "--engine",
        choices=["pdflatex", "xelatex", "lualatex", "latexmk"],
        default="latexmk",
        help="LaTeX engine to use (default: latexmk).",
    )
    parser.add_argument(
        "--bibtex",
        action="store_true",
        help="Run BibTeX after the first compilation (ignored with latexmk).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (aux, log, pdf, etc.).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove auxiliary files after successful compilation (latexmk -c).",
    )

    args = parser.parse_args()

    if args.engine != "latexmk" and args.bibtex:
        print("Multi-step compilation: pdflatex → bibtex → pdflatex × 2")
        tex_path = Path(args.texfile).resolve()
        base = tex_path.stem
        out_dir = args.output_dir or "."

        def run(cmd, step_name):
            print(f"\n--- {step_name} ---")
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                print(f"{step_name} failed.", file=sys.stderr)
                sys.exit(result.returncode)

        run(
            [args.engine, "-interaction=nonstopmode", "-halt-on-error"]
            + (["-output-directory", out_dir] if out_dir else [])
            + [str(tex_path)],
            "First pdflatex",
        )
        aux_path = Path(out_dir) / base if out_dir else Path(base)
        run(["bibtex", str(aux_path)], "BibTeX")
        # 3 & 4. Two more pdflatex runs
        for i in range(2):
            run(
                [args.engine, "-interaction=nonstopmode", "-halt-on-error"]
                + (["-output-directory", out_dir] if out_dir else [])
                + [str(tex_path)],
                f"pdflatex run {i+2}",
            )
        if args.clean:
            print("Cleaning auxiliary files...")
            for ext in [".aux", ".log", ".out", ".toc", ".bbl", ".blg"]:
                f = Path(out_dir) / (base + ext)
                if f.exists():
                    f.unlink()
        print("Multi-step compilation finished.")
        sys.exit(0)

    return_code = compile_tex(
        tex_file=args.texfile,
        engine=args.engine,
        bibtex=args.bibtex,
        output_dir=args.output_dir,
        clean=args.clean,
    )
    sys.exit(return_code)


if __name__ == "__main__":
    main()