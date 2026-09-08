#!/usr/bin/env python3
"""Supervise one local heavy task and stop only that task on host pressure."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple


EXIT_UNAVAILABLE = 69
EXIT_SAFETY_STOP = 70


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _mem_available_mib() -> int:
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _load_snapshot() -> Tuple[float, int, int]:
    with open("/proc/loadavg", encoding="ascii") as stream:
        fields = stream.read().split()
    running, total = fields[3].split("/", 1)
    return float(fields[0]), int(running), int(total)


def _psi_avg10(resource: str) -> float:
    try:
        with open("/proc/pressure/{}".format(resource), encoding="ascii") as stream:
            for line in stream:
                if line.startswith("some "):
                    for field in line.split()[1:]:
                        if field.startswith("avg10="):
                            return float(field.split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return 0.0


def _process_table() -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    parents: Dict[int, int] = {}
    ticks: Dict[int, int] = {}
    rss_kib: Dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            tail = stat[stat.rfind(")") + 2 :].split()
            parents[pid] = int(tail[1])
            ticks[pid] = int(tail[11]) + int(tail[12])
            with (entry / "status").open(encoding="ascii") as stream:
                for line in stream:
                    if line.startswith("VmRSS:"):
                        rss_kib[pid] = int(line.split()[1])
                        break
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError, IndexError):
            continue
    return parents, ticks, rss_kib


def _descendants(root: int, parents: Dict[int, int]) -> Set[int]:
    children: Dict[int, List[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    result: Set[int] = set()
    pending = [root]
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        result.add(pid)
        pending.extend(children.get(pid, ()))
    return result


def _tree_usage(root: int) -> Tuple[int, int, int]:
    parents, ticks, rss_kib = _process_table()
    members = _descendants(root, parents)
    return (
        sum(rss_kib.get(pid, 0) for pid in members) // 1024,
        sum(ticks.get(pid, 0) for pid in members),
        len(members),
    )


class _Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, **values: object) -> None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        fields = " ".join("{}={}".format(key, value) for key, value in sorted(values.items()))
        line = "{} event={} {}".format(timestamp, event, fields).rstrip()
        print(line, file=self._stream, flush=True)
        print(line, file=sys.stderr, flush=True)

    def close(self) -> None:
        self._stream.close()


def _terminate_group(process: subprocess.Popen[bytes], logger: _Logger, reason: str) -> None:
    if process.poll() is not None:
        return
    logger.write("terminate", pid=process.pid, reason=reason, signal="TERM")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if process.poll() is None:
        logger.write("terminate", pid=process.pid, reason=reason, signal="KILL")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _child_setup(cpus: Set[int]) -> None:
    os.setsid()
    os.sched_setaffinity(0, cpus)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--max-cpus", required=True, type=_positive_int)
    parser.add_argument("--min-start-available-mib", required=True, type=_positive_int)
    parser.add_argument("--safety-floor-mib", required=True, type=_positive_int)
    parser.add_argument("--sample-seconds", type=_positive_float, default=2.0)
    parser.add_argument("--low-memory-samples", type=_positive_int, default=3)
    parser.add_argument("--max-load-per-cpu", type=_positive_float, default=1.5)
    parser.add_argument("--cpu-pressure-percent", type=_positive_float, default=90.0)
    parser.add_argument("--cpu-overload-samples", type=_positive_int, default=15)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.safety_floor_mib >= args.min_start_available_mib:
        parser.error("safety floor must be below start admission threshold")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    logger = _Logger(Path(args.log_file).resolve())
    process: Optional[subprocess.Popen[bytes]] = None
    forwarded_signal: Optional[int] = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        if process is not None:
            _terminate_group(process, logger, "supervisor_signal_{}".format(signum))

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward)

    try:
        available = _mem_available_mib()
        allowed = sorted(os.sched_getaffinity(0))
        selected = set(allowed[: min(args.max_cpus, len(allowed))])
        if not selected:
            logger.write("admission_reject", reason="no_allowed_cpus")
            return EXIT_UNAVAILABLE
        if available < args.min_start_available_mib:
            logger.write(
                "admission_reject", available_mib=available,
                required_mib=args.min_start_available_mib,
            )
            return EXIT_UNAVAILABLE
        environment = dict(os.environ)
        threads = str(len(selected))
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "MAX_JOBS",
            "CMAKE_BUILD_PARALLEL_LEVEL", "PYTEST_XDIST_AUTO_NUM_WORKERS",
        ):
            environment[name] = threads
        logger.write(
            "start", available_mib=available, command=args.command[0],
            cpu_affinity=",".join(str(cpu) for cpu in sorted(selected)),
            max_cpus=len(selected), safety_floor_mib=args.safety_floor_mib,
        )
        process = subprocess.Popen(
            args.command,
            env=environment,
            preexec_fn=lambda: _child_setup(selected),
        )
        clock_ticks = os.sysconf("SC_CLK_TCK")
        previous_time = time.monotonic()
        _rss, previous_ticks, _members = _tree_usage(process.pid)
        low_memory = 0
        cpu_overload = 0
        while process.poll() is None:
            time.sleep(args.sample_seconds)
            now = time.monotonic()
            available = _mem_available_mib()
            load1, runnable, processes = _load_snapshot()
            memory_psi = _psi_avg10("memory")
            cpu_psi = _psi_avg10("cpu")
            rss_mib, current_ticks, members = _tree_usage(process.pid)
            elapsed = max(now - previous_time, 1e-6)
            cpu_percent = max(0.0, current_ticks - previous_ticks) / clock_ticks / elapsed
            cpu_percent = cpu_percent * 100.0 / len(selected)
            logger.write(
                "sample", available_mib=available, child_cpu_capacity_percent="{:.1f}".format(cpu_percent),
                child_processes=members, child_rss_mib=rss_mib, cpu_psi_avg10="{:.2f}".format(cpu_psi),
                load1="{:.2f}".format(load1), memory_psi_avg10="{:.2f}".format(memory_psi),
                runnable=runnable, system_processes=processes,
            )
            previous_time, previous_ticks = now, current_ticks
            low_memory = low_memory + 1 if available < args.safety_floor_mib else 0
            system_cpus = os.cpu_count() or len(allowed)
            overloaded = (
                load1 > system_cpus * args.max_load_per_cpu
                and runnable > system_cpus
                and cpu_psi >= args.cpu_pressure_percent
            )
            cpu_overload = cpu_overload + 1 if overloaded else 0
            if low_memory >= args.low_memory_samples:
                _terminate_group(process, logger, "low_memory")
                process.wait()
                logger.write("safety_stop", reason="low_memory", returncode=process.returncode)
                return EXIT_SAFETY_STOP
            if cpu_overload >= args.cpu_overload_samples:
                _terminate_group(process, logger, "sustained_cpu_pressure")
                process.wait()
                logger.write("safety_stop", reason="sustained_cpu_pressure", returncode=process.returncode)
                return EXIT_SAFETY_STOP
        raw_returncode = process.wait()
        returncode = 128 - raw_returncode if raw_returncode < 0 else raw_returncode
        logger.write("finish", raw_returncode=raw_returncode, returncode=returncode)
        if forwarded_signal is not None and returncode == 0:
            return 128 + forwarded_signal
        return returncode
    except (OSError, RuntimeError, ValueError) as exc:
        if process is not None:
            _terminate_group(process, logger, "supervisor_error")
        logger.write("supervisor_error", error=repr(exc))
        return EXIT_UNAVAILABLE
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
