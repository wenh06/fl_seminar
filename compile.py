"""
"""

import os
import shutil
import subprocess
import collections
import time
import signal
import platform
import argparse
import datetime
import shlex
from logging import Logger
from numbers import Real
from typing import Union, Optional, List, Tuple, NoReturn


def execute_cmd(cmd:Union[str,List[str]],
                timeout_hour:Real=0.1, 
                quiet:bool=False,
                logger:Optional[Logger]=None,
                raise_error:bool=True) -> Tuple[int, List[str]]:
    """
    """
    # cmd, shell_arg, executable_arg = _normalize_cmd_args(cmd)
    # if logger:
    #     logger.info("cmd = {}\nshell_arg = {}\nexecutable_arg = {}".format(cmd, shell_arg, executable_arg))
    shell_arg, executable_arg = False, None
    s = subprocess.Popen(
        cmd,
        shell=shell_arg,
        executable=executable_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=(not (platform.system().lower()=="windows")),
    )
    debug_stdout = collections.deque(maxlen=1000)
    msg = _str_center("  execute_cmd starts  ", 60)
    if logger:
        logger.info(msg)
    else:
        print(msg)
    start = time.time()
    now = time.time()
    timeout_sec = timeout_hour * 3600 if timeout_hour > 0 else float("inf")
    while now - start < timeout_sec:
        line = s.stdout.readline().decode("utf-8", errors="replace")
        if line.rstrip():
            debug_stdout.append(line)
            if logger:
                logger.debug(line)
            elif not quiet:
                print(line)
        exitcode = s.poll()
        if exitcode is not None:
            for line in s.stdout:
                debug_stdout.append(line.decode("utf-8", errors="replace"))
            if exitcode is not None and exitcode != 0:
                error_msg = " ".join(cmd) if not isinstance(cmd, str) else cmd
                error_msg += "\n"
                error_msg += "".join(debug_stdout)
                s.communicate()
                s.stdout.close()
                msg = _str_center("  execute_cmd failed  ", 60)
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                if raise_error:
                    raise subprocess.CalledProcessError(exitcode, error_msg)
                else:
                    output_msg = list(debug_stdout)
                    return exitcode, output_msg
            else:
                break
        now = time.time()
    # s.communicate()
    # s.terminate()
    s.kill()
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    # os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    s.stdout.close()
    output_msg = list(debug_stdout)

    msg = _str_center("  execute_cmd succeeded  ", 60)
    if logger:
        logger.info(msg)
    else:
        print(msg)

    exitcode = 0

    return exitcode, output_msg


def _str_center(s:str, length:int):
    """
    """
    return s.center(length, "*").center(length+2, "\n")


def get_parser() -> dict:
    """
    """
    description = "arguments to compile a tex project"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--compiler", type=str, default="xe",
        help=f"compiler, `xe` for xelatex, `pdf` for pdflatex, `lua` for lualatex, "
             "`dvi` for dvi, `ps` for postscript, `ps2pdf` for ps2pdf, `pdfdvi` for dvipdf",
        dest="compiler",
    )
    parser.add_argument(
        "-m", "--main", type=str, default="main.tex",
        help=f"filename of the main document (the compile entry)",
        dest="main",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help=f"filename of the output file",
        dest="output",
    )
    parser.add_argument(
        "-t", "--timeout", type=float, default=0.1,
        help=f"maximum running time of pm, in hours",
        dest="timeout_hour",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help=f"running quietly",
        dest="quiet",
    )

    args = vars(parser.parse_args())

    return args


def run(compiler:str, main:str, output:str, quiet:bool=False, timeout_hour:Real=0.1):
    cwd = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(cwd, "tmp_build")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(os.path.join(build_dir, "tikz"), exist_ok=True)
    fmt = {
        "xe": "pdfxe",
        "pdf": "pdf",
        "lua": "pdflua",
        "dvi": "dvi",
        "ps": "ps",
        "ps2pdf": "ps2pdf",
        "pdfdvi": "pdfdvi",
    }
    ext = {
        "xe": "pdf",
        "pdf": "pdf",
        "lua": "pdf",
        "dvi": "dvi",
        "ps": "ps",
        "ps2pdf": "pdf",
        "pdfdvi": "pdf",
    }
    cmd = " ".join([
        "latexmk",
        "-cd",
        "-f",
        f"-jobname={output}",
        f"-auxdir={build_dir}",
        f"-outdir={build_dir}",
        "-synctex=1",
        "-interaction=batchmode",
        f"-{fmt[compiler]}",
        f"{os.path.join(cwd, main)}",
    ])
    if not quiet:
        print(f"cmd = {cmd}")
    cmd = shlex.split(cmd)
    if not quiet:
        print(f"cmd = {cmd}")
    execute_cmd(cmd, timeout_hour=timeout_hour, quiet=quiet, raise_error=False)
    output_filename = f"{output}.{ext[compiler]}"
    os.rename(os.path.join(build_dir, output_filename), os.path.join(cwd, output_filename))
    print(f"Compilation finishes and output file `{output_filename}` is produced.")
    print("ignore the ending message from `execute_cmd`.")
    shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    args = get_parser()
    print(f"args = {args}")
    run(
        compiler=args.get("compiler"),
        main=args.get("main"),
        output=args.get("output"),
        quiet=args.get("quiet"),
        timeout_hour=args.get("timeout_hour"),
    )
    exit(0)
