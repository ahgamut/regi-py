from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys
import os
import platform


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


def get_src_cpp(path):
    cpps = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[-1] == ".cc":
                cpps.append(os.path.join(root, filename))
    return cpps


# extension naming matters, because
# 1) the .DEF file used by the linker on windows
# 2) the copying from build/ to dist/
ext_modules = [
    Extension(
        name="regi_py.core",
        sources=get_src_cpp("src/regi_py/core/"),
        include_dirs=[
            str(get_pybind_include(True)),
            str(get_pybind_include(False)),
            "src/regi_py/core",
        ],
        extra_compile_args=["-DUSE_UNICODE"],
        extra_link_args=[],
        language="c++",
    )
]


class BuildExt(_build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": ["-Wall", "-Wpedantic", "-Wno-unused-result", "-std=c++17"],
    }

    if platform.system() == "Windows":
        os.environ["CC"] = "msvc"
        os.environ["CXX"] = "msvc"

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct)
        if ct == "unix":
            opts.append("-fvisibility=hidden")
            if self.compiler.compiler_so:
                if "-g" in self.compiler.compiler_so:
                    self.compiler.compiler_so.remove("-g")
        elif ct == "mingw32":
            opts = self.c_opts.get("unix")

        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args + opts
        _build_ext.build_extensions(self)


setup(
    name="regi_py",
    version="0.0.1",
    author="Gautham Venkatasubramanian",
    author_email="ahgamut@gmail.com",
    description="implementation of game mechanics of regicide",
    license="MIT",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ahgamut/regi-py",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False,
    install_requires=["pybind11>=3"],
    setup_requires=["pybind11>=3"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
