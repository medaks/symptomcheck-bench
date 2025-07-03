import subprocess
import shlex
from logging import getLogger
from typing import Optional

logger = getLogger("medask.util.bash")


def exec_bg(cmd: str, shell: bool = False) -> None:
    """Execute a command that will run in the background."""
    logger.info(f"{cmd}")
    if shell is False:
        # Split cmd into list of strings.
        cmd = shlex.split(cmd)  # type: ignore
    subprocess.Popen(cmd)


def exec(
    cmd: str,
    check: bool = True,
    shell: bool = True,
    timeout: Optional[float] = None,
    log: bool = True,
) -> str:
    """Execute a command, return STDOUT."""
    if log:
        logger.info(f"{cmd}")
    if shell is False:
        # Split cmd into list of strings.
        cmd = shlex.split(cmd)  # type: ignore
    res = subprocess.run(
        cmd,
        shell=shell,
        check=check,  # Check for errors.
        text=True,  # Decode output to unicode (which python uses).
        stdout=subprocess.PIPE,  # Capture STDOUT.
        timeout=timeout,  # Timeout for the command.
    )
    return res.stdout
