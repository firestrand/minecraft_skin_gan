"""Build and test a fresh, non-editable install, independent of the checkout."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    uv = root / "scripts" / "uv.sh"
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    with tempfile.TemporaryDirectory(prefix="minecraft-skin-fresh-") as directory:
        fresh = Path(directory)
        for name in (
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "requirements-gpu.txt",
            "README.md",
            "LICENSE",
        ):
            shutil.copy2(root / name, fresh / name)
        for module in root.glob("*.py"):
            shutil.copy2(module, fresh / module.name)
        for name in ("tests", "scripts", "docs", "minecraft_skin_gan"):
            shutil.copytree(root / name, fresh / name, ignore=shutil.ignore_patterns("__pycache__"))
        (fresh / "artifacts").mkdir()
        subprocess.run(
            [uv, "sync", "--locked", "--group", "dev", "--no-editable"],
            cwd=fresh,
            env=env,
            check=True,
        )
        subprocess.run(
            [uv, "build", "--no-sources", "--no-build-isolation"], cwd=fresh, env=env, check=True
        )
        wheel = next((fresh / "dist").glob("*.whl"))
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                fresh / ".venv/bin/python",
                "--no-deps",
                "--reinstall",
                wheel,
            ],
            cwd=fresh,
            env=env,
            check=True,
        )
        # An external cwd proves imports resolve from the wheel rather than source files.
        outside = fresh / "outside"
        outside.mkdir()
        subprocess.run(
            [fresh / ".venv/bin/skin-gan", "--help"],
            cwd=outside,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                fresh / ".venv/bin/python",
                "-c",
                "import pathlib, create_skins_array, download_skins, generate_skin, remove_duplicates, simple_gan, sort_skins; assert 'site-packages' in str(pathlib.Path(simple_gan.__file__)); print('Installed wheel imports passed')",
            ],
            cwd=outside,
            env=env,
            check=True,
        )
        subprocess.run(
            [uv, "run", "--locked", "--no-editable", "pytest"], cwd=fresh, env=env, check=True
        )
        subprocess.run(
            [uv, "run", "--locked", "--no-editable", "python", "scripts/check_coverage.py"],
            cwd=fresh,
            env=env,
            check=True,
        )
    print("Fresh locked install, wheel imports, tests and coverage passed")


if __name__ == "__main__":
    main()
