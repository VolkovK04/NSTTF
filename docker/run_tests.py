import glob
import itertools
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

_ROOT_PATH = os.path.abspath(  # script parent directory
    os.path.dirname(os.path.dirname(__file__))  # script directory
)


def _report_fatal_error(msg: str):
    """
    Dump error message and stop script execution.
    """
    print(f"  [FAILED] {msg}")
    sys.exit(1)


def _exists(path: str) -> bool:
    return os.path.exists(path)


class StdCppLib(Enum):
    GnuLibStdCpp = auto()
    ClangLibCpp = auto()


@dataclass
class ToolSet:
    name: str
    cpp_compiler: str
    c_compiler: str


@dataclass(frozen=True)
class TestingConfiguration:
    build_dir: str
    toolset: ToolSet
    std_cpp_lib: StdCppLib
    build_type: str
    cpp_flags: str


__TOOL_SETS = [
    ToolSet(name="gcc_12", cpp_compiler="g++-12", c_compiler="gcc-12"),
    # ToolSet(name="clang_15", cpp_compiler="clang++-15", c_compiler="clang-15"),
]


def _get_cpp_flags(build_type: str) -> str:
    flags = []
    # flags = ["-Wall", "-Werror"]

    if build_type == "Debug":
        flags += ["-O0", "-g"]
    elif build_type == "Release":
        flags += ["-O2"]
    else:
        assert False, f"unexpected build type {build_type}"

    return " ".join(flags)


def _get_cpp_flags_asan(toolset: ToolSet, build_type: str) -> str:
    std_flags = _get_cpp_flags(build_type)

    if toolset.name.startswith("gcc"):
        flags = ["-fsanitize=address", "-g", "-fno-sanitize-recover"]
    elif toolset.name.startswith("clang"):
        flags = [
            "-fsanitize=address",
            "-g",
            "-fno-omit-frame-pointer",
            "-fno-optimize-sibling-calls",
        ]
    else:
        assert False, f"unexpected tool set {toolset}"

    flags = " ".join(flags)
    return f"{std_flags} {flags}"


def _get_cpp_flags_usan(toolset: ToolSet, build_type: str) -> str:
    std_flags = _get_cpp_flags(build_type)

    if toolset.name.startswith("gcc"):
        flags = ["-fsanitize=undefined", "-g", "-fno-sanitize-recover"]
    elif toolset.name.startswith("clang"):
        flags = ["-fsanitize=undefined", "-g", "-fno-omit-frame-pointer"]
    else:
        assert False, f"unexpected tool set {toolset}"

    flags = " ".join(flags)
    return f"{std_flags} {flags}"


def _make_full_build_path(
    toolset: ToolSet, build_type: str, san: str, std_lib: StdCppLib
) -> str:
    if std_lib == StdCppLib.GnuLibStdCpp:
        std_lib_str = "libstdcpp"
    elif std_lib == StdCppLib.ClangLibCpp:
        std_lib_str = "libcpp"
    else:
        assert False, f"unexpected std lib {std_lib}"

    return os.path.join(
        _ROOT_PATH, f"build_{toolset.name}_{build_type}_{san}_{std_lib_str}"
    )


def _get_supported_std_cpp_libs(toolset: ToolSet) -> List[StdCppLib]:
    if toolset.name.startswith("gcc"):
        return [StdCppLib.GnuLibStdCpp]

    if toolset.name.startswith("clang"):
        return [StdCppLib.GnuLibStdCpp, StdCppLib.ClangLibCpp]

    assert False, f"unexpected toolset: {toolset}"


def _make_testing_configurations() -> List[TestingConfiguration]:
    rv = []
    for toolset in __TOOL_SETS:
        for build_type, std_lib in itertools.product(  # build types.
            [
                "Debug", 
            #  "Release"
             ],  # cpp standard libs
            _get_supported_std_cpp_libs(toolset),
        ):

            def make_build_dir(san: str):
                return _make_full_build_path(
                    toolset=toolset,
                    build_type=build_type,
                    san=san,
                    std_lib=std_lib,
                )

            rv += [
                # TestingConfiguration(
                #     build_dir=make_build_dir("orig"),
                #     toolset=toolset,
                #     std_cpp_lib=std_lib,
                #     build_type=build_type,
                #     cpp_flags=_get_cpp_flags(build_type),
                # ),
                TestingConfiguration(
                    build_dir=make_build_dir("asan"),
                    toolset=toolset,
                    std_cpp_lib=std_lib,
                    build_type=build_type,
                    cpp_flags=_get_cpp_flags_asan(toolset, build_type),
                ),
                # TestingConfiguration(
                #     build_dir=make_build_dir("usan"),
                #     toolset=toolset,
                #     std_cpp_lib=std_lib,
                #     build_type=build_type,
                #     cpp_flags=_get_cpp_flags_usan(toolset, build_type),
                # ),
            ]

    # Ensure that each test configuration build directory is unique.
    if len(set(cfg.build_dir for cfg in rv)) != len(rv):
        _report_fatal_error("non-unique build dirs in testing configurations")

    return rv


def _build(cfg: TestingConfiguration):
    print(f"\nBuilding: {cfg.build_dir}")

    if _exists(cfg.build_dir):
        shutil.rmtree(cfg.build_dir)
    os.makedirs(cfg.build_dir, exist_ok=False)
    os.chdir(cfg.build_dir)

    result = subprocess.run(
        [
            "cmake",
            "-G",
            "Ninja",
            f"-DCMAKE_BUILD_TYPE={cfg.build_type}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            _ROOT_PATH,
        ],
        env=dict(
            CC=cfg.toolset.c_compiler,
            CXX=cfg.toolset.cpp_compiler,
            CXXFLAGS=cfg.cpp_flags,
            **os.environ,
        ),
    )
    if result.returncode:
        _report_fatal_error(f"Failed to generate cmake project {cfg.build_dir}")

    result = subprocess.run(["cmake", "--build", "."])
    if result.returncode:
        _report_fatal_error(f"Failed to build {cfg.build_dir}")


def _run_unit_tests(cfg: TestingConfiguration):
    print(f"\nRun unit tests: {cfg.build_dir}")

    assert _exists(cfg.build_dir), f"must exist {cfg.build_dir}"

    os.chdir(cfg.build_dir)

    if subprocess.run(["./tests/NSTTF_test"]).returncode:
        _report_fatal_error(f"Unit tests failed for {cfg.build_dir}")


@dataclass(frozen=True)
class IntegrationTestCase:
    scenario: str
    is_positive: bool
    expected_output: Optional[str]


def _run_static_analyzers(configurations: List[TestingConfiguration]):
    os.chdir(_ROOT_PATH)

    print("\nRun cppcheck:")
    if subprocess.run(["cppcheck", "src", "--error-exitcode=2"]).returncode:
        _report_fatal_error("CppCheck has detected several issues.")

    # for cfg in configurations:
    #     print(f"\nRun clang-tidy for {cfg.build_dir}")
    #     result = subprocess.run(["clang-tidy-15", "-p", cfg.build_dir, "**/*.cpp"])
    #     if result.returncode:
    #         _report_fatal_error('clang-tidy has detected several issues.')


def _check_format():
    print(f"\nCheck clang-format")
    os.chdir(_ROOT_PATH)

    files = (
        list(glob.glob("src/**/*.cpp", recursive=True))
        + list(glob.glob("src/**/*.cxx", recursive=True))
        + list(glob.glob("src/**/*.c", recursive=True))
        + list(glob.glob("src/**/*.h", recursive=True))
        + list(glob.glob("src/**/*.hxx", recursive=True))
    )
    for f in files:
        print(f"Check {f}")
        if subprocess.run(["clang-format-15", f, "--dry-run", "-Werror"]).returncode:
            _report_fatal_error(f"Run clang-format on {f}")


def _cleanup(cfg: TestingConfiguration) -> None:
    if os.path.exists(cfg.build_dir):
        shutil.rmtree(cfg.build_dir)


def _main():
    _check_format()

    configurations = _make_testing_configurations()

    for cfg in configurations[:1]:
        _build(cfg)

    _run_static_analyzers(configurations)

    for cfg in configurations[:1]:
        _run_unit_tests(cfg)

    # TODO: launch valgrind

    for cfg in configurations[:1]:
        _cleanup(cfg)

    print("\nSuccess")


if __name__ == "__main__":
    _main()
