"""Console entry point, so an installed copy has a command of its own."""
import os
import runpy
import sys
from pathlib import Path


def _find_runner():
    """run_all.py, either beside the package or under PROJECT_ROOT."""
    here = Path(__file__).resolve().parent.parent / "run_all.py"
    if here.exists():
        return here
    root = os.environ.get("PROJECT_ROOT")
    if root:
        cand = Path(root) / "run_all.py"
        if cand.exists():
            return cand
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--root" in argv:
        i = argv.index("--root")
        try:
            os.environ["PROJECT_ROOT"] = str(Path(argv[i + 1]).resolve())
        except IndexError:
            sys.exit("--root needs a path")
        del argv[i:i + 2]

    runner = _find_runner()
    if runner is None:
        sys.exit("cannot find run_all.py; pass --root <project directory> "
                 "or set PROJECT_ROOT")

    sys.argv = [str(runner)] + argv
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
