"""Optional C++17 additive-TPL accelerator; build with the preparation interpreter."""
from pathlib import Path
import os
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / "code")
setup(name="myst-grounded", version="1.0", ext_modules=[Extension(
    "grounded_chain_native", [str(ROOT / "code/native/grounded_chain_retriever.cpp")], language="c++",
    extra_compile_args=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"])])
