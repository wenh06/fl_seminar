"""
"""

import collections
import os
import shutil
import subprocess
import platform
import tempfile
import tarfile
import zipfile
from pathlib import Path
from typing import NoReturn, Union, List, Tuple, Optional

import requests
import tqdm

from misc import CACHED_DATA_DIR


__all__ = [
    "download_if_needed",
]


FEDML_DOMAIN = "https://fedml.s3-us-west-1.amazonaws.com/"
DOWNLOAD_CMD = "wget --no-check-certificate --no-proxy {url} -O {dst}"
DECOMPRESS_CMD = {
    "tar": "tar -xvf {src} --directory {dst_dir}",
    "zip": "unzip {src} -d {dst_dir}",
}


def download_if_needed(
    url: str, dst_dir: Union[str, Path] = CACHED_DATA_DIR, extract: bool = True
) -> NoReturn:
    """ """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() and len(list(dst_dir.iterdir())) > 0:
        return
    http_get(url, dst_dir, extract=extract)


def http_get(
    url: str,
    dst_dir: Union[str, Path],
    proxies: Optional[dict] = None,
    extract: bool = True,
) -> NoReturn:
    """Get contents of a URL and save to a file.

    https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    print(f"Downloading {url}.")
    parent_dir = Path(dst_dir).parent
    downloaded_file = tempfile.NamedTemporaryFile(
        dir=parent_dir, suffix=_suffix(url), delete=False
    )
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403 or req.status_code == 404:
        raise Exception(f"Could not reach {url}.")
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            downloaded_file.write(chunk)
    progress.close()
    downloaded_file.close()
    if extract:
        if _suffix(url) == ".zip":
            _unzip_file(str(downloaded_file.name), str(dst_dir))
        else:  # tar files
            _untar_file(str(downloaded_file.name), str(dst_dir))
    else:
        shutil.copyfile(downloaded_file.name, Path(dst_dir) / Path(url).name)
    os.remove(downloaded_file.name)


def execute_cmd(
    cmd: Union[str, List[str]], raise_error: bool = True
) -> Tuple[int, List[str]]:
    """ """
    shell_arg, executable_arg = False, None
    s = subprocess.Popen(
        cmd,
        shell=shell_arg,
        executable=executable_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=(not (platform.system().lower() == "windows")),
    )
    debug_stdout = collections.deque(maxlen=1000)
    while True:
        line = s.stdout.readline().decode("utf-8", errors="replace")
        if line.rstrip():
            debug_stdout.append(line)
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
                if raise_error:
                    raise subprocess.CalledProcessError(exitcode, error_msg)
                else:
                    output_msg = list(debug_stdout)
                    return exitcode, output_msg
            else:
                break
    s.communicate()
    # s.terminate()
    # s.kill()
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
    # os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    s.stdout.close()
    output_msg = list(debug_stdout)

    exitcode = 0

    return exitcode, output_msg


def _stem(path: Union[str, Path]) -> str:
    """ """
    ret = Path(path).stem
    for _ in range(3):
        ret = Path(ret).stem
    return ret


def _suffix(path: Union[str, Path]) -> str:
    """ """
    return "".join(Path(path).suffixes)


def _unzip_file(
    path_to_zip_file: Union[str, Path], dst_dir: Union[str, Path]
) -> NoReturn:
    """Unzips a .zip file to folder path."""
    print(f"Extracting file {path_to_zip_file} to {dst_dir}.")
    with zipfile.ZipFile(str(path_to_zip_file)) as zip_ref:
        zip_ref.extractall(str(dst_dir))


def _untar_file(
    path_to_tar_file: Union[str, Path], dst_dir: Union[str, Path]
) -> NoReturn:
    """decompress a .tar.xx file to folder path."""
    print(f"Extracting file {path_to_tar_file} to {dst_dir}.")
    mode = Path(path_to_tar_file).suffix.replace(".", "r:").replace("tar", "")
    # print(f"mode: {mode}")
    with tarfile.open(str(path_to_tar_file), mode) as tar_ref:
        tar_ref.extractall(str(dst_dir))
