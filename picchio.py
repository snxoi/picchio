#!/usr/bin/env python3
# picchio: knocks on your local LLM setup and listens for hollow spots.
#
# What it does, in one run:
#   1. runs the same fixed prompt through your model N times (default 3;
#      the first pass is the cold one, the rest are warm)
#   2. reads the engine's own timing and placement evidence, and on
#      macOS also samples the OS's own GPU meter (ioreg) while it runs
#   3. reports prefill, decode and wallclock tok/s as three separate lanes,
#      cold pass first, then the warm median and the warm spread
#   4. tells you whether the GPU actually did the work, or quietly did
#      not: the engine's claim, the OS meter and the speed signature
#      must agree, and any two of them fighting is its own verdict;
#      on a degraded verdict one WHY line names the cause it can prove
#      (explicit flag, memory fit, init failure) or says unknown
#   5. shows where the seconds of the cold pass went (load, prefill, decode)
#   6. prints a verdict block sized to fit in a forum comment
#
# Usage:
#   python3 picchio.py /path/to/model.gguf            llama.cpp, full diagnosis
#   python3 picchio.py qwen3.5:9b                     ollama tag, measurement
#   python3 picchio.py http://127.0.0.1:8080          llama-server endpoint,
#                                       measurement of a server already up
#   python3 picchio.py model.gguf --explain 36        classify a number you saw
#   python3 picchio.py --explain 36                   same, against last run
#   python3 picchio.py --selftest                     replay examples/raw
#   python3 picchio.py model.gguf -- --device none -ngl 0
#                                       (args after -- go to the engine)
#   python3 picchio.py guard -- llama-server --verbose -m model.gguf
#                                       (watch your own command; warn on
#                                        degraded placement, never kill)
#   python3 picchio.py compare mine.txt theirs.txt
#                                       (diff two pasted verdict blocks,
#                                        name the variable that did it)
#   python3 picchio.py verify block.txt
#                                       (re-derive a pasted block's own
#                                        physics; flag it if it lies)
#   python3 picchio.py watch --engine ollama
#                                       (point the OS gpu meter at any
#                                        running engine; no stderr parse)
#   python3 picchio.py monitor http://127.0.0.1:8080
#                                       (probe a running server on a timer;
#                                        flag any request the GPU dropped)
#   python3 picchio.py model.gguf --ctx-sweep
#                                       (re-measure at 4k/16k/32k context
#                                        and report the decode decay slope)
#   python3 picchio.py plan [MODEL]
#                                       (will it fit, from the gguf header;
#                                        decode estimate once calibrated)
#
# Needs: python3 (any recent one), plus llama.cpp on PATH or a local ollama.
# Nothing else. No pip.
#
# Exit codes: 0 ok/healthy, 2 could not run, 3 partial offload,
#             4 silent cpu fallback, 5 conflicting evidence.
#             verify: 0 self-consistent, 5 flagged, 2 unreadable.

import argparse
import ctypes
import glob
import io
import json
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "0.1.0"
# Measurement protocol tag, printed in the block footer. If the prompt
# size, generation length, pass structure or aggregation ever change,
# this bumps, so numbers from different protocols never get compared as
# if they were one series.
PROTOCOL = "mp1"
WIDTH = 66
N_PREDICT = 128
CTX = 4096
CACHE_PATH = os.path.expanduser("~/.cache/picchio/last.json")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")

# A fixed prompt of roughly 730 tokens. Short prompts lie: 7 prompt
# tokens measured 36 tok/s of apparent prefill on the same setup where
# 730 tokens measured about 590, because per call overhead dominates
# below a few hundred tokens. 128 generated tokens because decode
# settles within the first few dozen and 128 gives the median room
# without stretching the run.
_PARA = (
    "A benchmark number without its measurement conditions is a rumor "
    "with digits in it. Tokens per second can describe how fast a model "
    "reads a prompt, how fast it writes an answer, or how long the whole "
    "exchange took including loading the weights from disk. These three "
    "rates differ by an order of magnitude on the same machine in the "
    "same minute, and none of them is wrong. What is wrong is quoting "
    "one of them without saying which one it is. "
)
BENCH_PROMPT = "".join(
    "Consider case number {}: {}".format(i + 1, _PARA) for i in range(8)
)


# ----------------------------------------------------------------- machine

def _cmd_out(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return ""


def machine_info():
    info = {"os": "", "chip": "", "ram_gb": None}
    sysname = platform.system()
    if sysname == "Darwin":
        info["chip"] = _cmd_out(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _cmd_out(["sysctl", "-n", "hw.memsize"])
        if mem.isdigit():
            info["ram_gb"] = round(int(mem) / (1024 ** 3))
        info["os"] = "macOS " + platform.mac_ver()[0]
    elif sysname == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        info["chip"] = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024 ** 2))
                        break
        except OSError:
            pass
        info["os"] = "Linux " + platform.release()
        try:
            # the gpu belongs in the machine fingerprint on linux; the
            # nvml name (display form) rides the chip field so every
            # footer and cache entry carries it without a new column
            gpu = _NVML().device_name()
            if gpu:
                info["chip"] = "{} + {}".format(info["chip"], gpu) \
                    if info["chip"] else gpu
        except Exception:
            pass
    else:
        info["os"] = sysname
    if not info["chip"]:
        info["chip"] = platform.machine() or "unknown cpu"
    return info


def blank_pass():
    return {
        "wall_s": None,
        "load_ms": None,
        "prompt_ms": None, "prompt_tokens": None,
        "eval_ms": None, "eval_tokens": None,
        "offload_n": None, "offload_total": None,
        "gpu_device": None, "gpu_kind": None,
        "model_params": None, "model_size": None, "model_bytes": None,
        "threads": None, "cores": None,
        "vram_frac": None, "n_expert": None,
        "kv_types": None, "tensor_types": None,
        "free_mib": None, "fit_seen": False, "init_fail": None,
        "prefill_toks": None, "decode_toks": None, "wallclock_toks": None,
    }


def finish_rates(d):
    if d["prompt_ms"] and d["prompt_tokens"]:
        d["prefill_toks"] = d["prompt_tokens"] / (d["prompt_ms"] / 1000.0)
    if d["eval_ms"] and d["eval_tokens"]:
        d["decode_toks"] = d["eval_tokens"] / (d["eval_ms"] / 1000.0)
    if d["eval_tokens"] and d["wall_s"]:
        d["wallclock_toks"] = d["eval_tokens"] / d["wall_s"]
    return d


def size_bytes(s):
    """'5.28 GiB' -> bytes, None when the unit is unfamiliar."""
    m = re.match(r"([\d.]+)\s*([KMG]i?B|B)", s or "", re.I)
    if not m:
        return None
    mult = {"b": 1, "kib": 1024, "kb": 1000, "mib": 1024 ** 2,
            "mb": 1000 ** 2, "gib": 1024 ** 3, "gb": 1000 ** 3}
    return int(float(m.group(1)) * mult[m.group(2).lower()])


def keep_log(path, text):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    except OSError as e:
        sys.stderr.write("picchio: could not write {}: {}\n".format(path, e))


# ------------------------------------------------------- engine: llama.cpp

def find_binary(explicit):
    if explicit:
        if shutil.which(explicit) or os.path.isfile(explicit):
            return explicit
        sys.exit("picchio: engine binary not found: {}".format(explicit))
    # llama-completion is the one-shot binary on current llama.cpp builds;
    # older builds did the same job with llama-cli -no-cnv.
    for name in ("llama-completion", "llama-cli"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit(
        "picchio: could not find llama-completion or llama-cli on PATH.\n"
        "Install llama.cpp (e.g. brew install llama.cpp) or pass --bin."
    )


def parse_engine_version(out):
    m = re.search(r"version:\s*(\S+)\s*\(([0-9a-f]+)\)", out)
    if m:
        return "b" + m.group(1)
    # tarball builds carry no git hash; take a bare version number when
    # --version still prints one. llama.cpp prints version: 0 (unknown)
    # on such builds, and that 0 is a sentinel, not a version: say
    # version unknown instead of dressing the sentinel up as build b0.
    m = re.search(r"version:\s*(\d\S*)", out)
    if m and m.group(1) != "0":
        return "b" + m.group(1).lstrip("b")
    return "(version unknown)"


def engine_version(binpath):
    return parse_engine_version(_cmd_out([binpath, "--version"]))


def run_llama_pass(binpath, model, extra_args, log_path=None,
                   prompt=BENCH_PROMPT, ctx=CTX):
    base = [
        binpath,
        "-m", model,
        "-p", prompt,
        "-n", str(N_PREDICT),
        "-c", str(ctx),
        "--seed", "7",
        "--ignore-eos",
    ]
    # Newest flags first; older builds reject flags they predate, so on
    # failure retry with a smaller flag set before giving up.
    attempts = [
        base + ["-no-cnv", "--verbose"],
        base + ["-no-cnv"],
        base,
    ]
    last = None
    for args in attempts:
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                args + extra_args,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            sys.exit("picchio: engine run exceeded 30 minutes, giving up.")
        wall_s = time.monotonic() - t0
        if r.returncode == 0:
            keep_log(log_path, r.stderr)
            return parse_stderr(r.stderr, wall_s)
        last = r
    tail = "\n".join(last.stderr.strip().splitlines()[-6:])
    sys.exit(
        "picchio: engine exited with code {}.\nLast lines:\n{}".format(
            last.returncode, tail
        )
    )


def parse_stderr(text, wall_s):
    d = blank_pass()
    d["wall_s"] = wall_s
    re_load = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
    re_pair = re.compile(r"=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)")
    re_off = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")
    re_metal = re.compile(r"ggml_metal_init: found device:\s*(.+)")
    re_cuda = re.compile(r"Device\s+\d+:\s*([^,]+),")
    # e.g. "using device CUDA0 (NVIDIA GeForce RTX 4090) (0000:61:00.0)
    # - 23818 MiB free": on b9430 CUDA builds this is the only line
    # naming the device (no ggml_cuda_init, no "Device 0:" lines exist
    # there, verified on the 4090 fixtures)
    re_cuda_dev = re.compile(r"using device CUDA\d+ \(([^)]+)\)")
    re_params = re.compile(r"model params\s*=\s*([\d.]+\s*\S?)")
    re_size = re.compile(r"file size\s*=\s*([\d.]+\s*\S+)")
    re_threads = re.compile(r"n_threads\s*=\s*(\d+).*?/\s*(\d+)")
    # e.g. "using device MTL0 (Apple M5) (unknown id) - 25558 MiB free":
    # the free figure the engine itself saw, kept for WHY attribution.
    re_free = re.compile(r"-\s*(\d+)\s*MiB free")
    # e.g. "llama_kv_cache: size = 4352.00 MiB (..., 1/1 seqs),
    # K (q8_0): 2176.00 MiB, V (q8_0): 2176.00 MiB": the runtime kv
    # dtype, measured here on b9430 with -ctk q8_0 -ctv q8_0 (the f16
    # default is in every committed fixture). Cached so the id card
    # can cite a dtype this machine has actually run.
    re_kvtypes = re.compile(r"llama_kv_cache: size =.*"
                            r"K \((\S+)\):.*V \((\S+)\):")
    # e.g. "llama_model_loader: - type q4_K:  132 tensors": the
    # loader's own per-type census, the engine side of the id cross
    # check against the gguf table walk
    re_ttype = re.compile(r"- type\s+(\S+):\s+(\d+) tensors")

    for line in text.splitlines():
        if "prompt eval time" in line:
            m = re_pair.search(line)
            if m:
                d["prompt_ms"] = float(m.group(1))
                d["prompt_tokens"] = int(m.group(2))
        elif "eval time" in line:
            m = re_pair.search(line)
            if m:
                d["eval_ms"] = float(m.group(1))
                d["eval_tokens"] = int(m.group(2))
        elif "load time" in line:
            m = re_load.search(line)
            if m:
                d["load_ms"] = float(m.group(1))
        m = re_off.search(line)
        if m:
            d["offload_n"] = int(m.group(1))
            d["offload_total"] = int(m.group(2))
        m = re_metal.search(line)
        if m:
            d["gpu_device"] = m.group(1).strip()
            d["gpu_kind"] = "Metal"
        m = re_cuda_dev.search(line)
        if m and not d["gpu_device"]:
            d["gpu_kind"] = "CUDA"
            # display form: the NVIDIA prefix drops so the gpu line
            # holds the 66 column budget; the raw line is in the log
            d["gpu_device"] = re.sub(r"^NVIDIA\s+", "",
                                     m.group(1).strip())
        if "ggml_cuda_init" in line or "CUDA devices" in line:
            d["gpu_kind"] = d["gpu_kind"] or "CUDA"
        m = re_cuda.search(line)
        if m and d["gpu_kind"] == "CUDA" and not d["gpu_device"]:
            d["gpu_device"] = m.group(1).strip()
        if "ggml_vulkan" in line.lower() and not d["gpu_kind"]:
            d["gpu_kind"] = "Vulkan"
        m = re_params.search(line)
        if m:
            d["model_params"] = m.group(1).strip()
        m = re_size.search(line)
        if m:
            d["model_size"] = m.group(1).strip()
            d["model_bytes"] = size_bytes(d["model_size"])
        if "system_info" in line:
            m = re_threads.search(line)
            if m:
                # llama.cpp defaults to 4 threads on this 10 core test
                # machine; recorded rather than tuned, because CPU rates
                # move a lot with -t and the block should say so.
                d["threads"] = int(m.group(1))
                d["cores"] = int(m.group(2))
        m = re_free.search(line)
        if m:
            d["free_mib"] = int(m.group(1))
        m = re_kvtypes.search(line)
        if m:
            d["kv_types"] = [m.group(1), m.group(2)]
        m = re_ttype.search(line)
        if m:
            # the loader prints the census once per load and loads
            # twice per pass; identical values, overwrite is idempotent
            d["tensor_types"] = d["tensor_types"] or {}
            d["tensor_types"][m.group(1)] = int(m.group(2))
        m = re.search(r"n_expert\s+=\s*(\d+)", line)
        if m:
            # 0 on a dense model; the cache keeps this so plan knows a
            # mixture of experts cannot calibrate bandwidth arithmetic
            d["n_expert"] = int(m.group(1))
        if "common_params_fit_impl" in line:
            d["fit_seen"] = True
        low = line.lower()
        if (d["init_fail"] is None
                and ("error" in low or "failed" in low)
                and ("ggml_metal" in low or "ggml_cuda" in low
                     or "ggml_vulkan" in low or "ggml_backend" in low)):
            # first backend init failure line, verbatim minus the
            # "0.00.061.339 I " style log prefix some builds prepend
            d["init_fail"] = re.sub(r"^[\d.]+\s+[A-Z]\s+", "",
                                    line.strip())
    return finish_rates(d)


# ---------------------------------------------------------- engine: ollama

def ollama_api(path, payload=None, timeout=1800):
    url = "http://{}{}".format(OLLAMA_HOST, path)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def ollama_reachable():
    try:
        return ollama_api("/api/version", timeout=3).get("version", "?")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def ollama_has_model(tag):
    try:
        ollama_api("/api/show", {"model": tag}, timeout=15)
        return True
    except urllib.error.HTTPError:
        return False


def ollama_ps_entry(tag):
    try:
        for m in ollama_api("/api/ps", timeout=15).get("models", []):
            if m.get("name") == tag or m.get("model") == tag:
                return m
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


def map_ollama(resp, wall_s, ps):
    d = blank_pass()
    d["wall_s"] = wall_s
    ns = 1e6  # ns -> ms
    if resp.get("load_duration"):
        d["load_ms"] = resp["load_duration"] / ns
    if resp.get("prompt_eval_duration") and resp.get("prompt_eval_count"):
        d["prompt_ms"] = resp["prompt_eval_duration"] / ns
        d["prompt_tokens"] = resp["prompt_eval_count"]
    if resp.get("eval_duration") and resp.get("eval_count"):
        d["eval_ms"] = resp["eval_duration"] / ns
        d["eval_tokens"] = resp["eval_count"]
    if ps:
        size, vram = ps.get("size"), ps.get("size_vram")
        if size:
            d["model_size"] = "{:.2f} GiB".format(size / (1024 ** 3))
            d["model_bytes"] = size
            d["vram_frac"] = (vram or 0) / size
        det = ps.get("details") or {}
        if det.get("parameter_size"):
            d["model_params"] = det["parameter_size"].rstrip("B") + " B"
        if det.get("quantization_level"):
            d["model_params"] += ", " + det["quantization_level"]
    return finish_rates(d)


def run_ollama_pass(tag, log_path=None, prompt=BENCH_PROMPT, ctx=CTX):
    t0 = time.monotonic()
    resp = ollama_api("/api/generate", {
        "model": tag,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": N_PREDICT, "num_ctx": ctx, "seed": 7},
    })
    wall_s = time.monotonic() - t0
    keep_log(log_path, json.dumps(resp, indent=1))
    ps = ollama_ps_entry(tag)
    return map_ollama(resp, wall_s, ps), ps


def looks_like_tag(s):
    """An ollama tag has no path separator and no .gguf suffix. Anything
    path shaped that does not exist on disk must be reported as a missing
    file, not quietly retried as a tag: a diagnostic that misdiagnoses
    its own arguments has no business diagnosing your GPU."""
    return "/" not in s and not s.lower().endswith(".gguf")


HINT_NO_MODELS = (
    "picchio: no model given, and none found in the usual places\n"
    "(no ollama tags, no .gguf in the current folder, the HF cache,\n"
    "or the LM Studio folders).\n\n"
    "Point it at any .gguf file or ollama tag:\n"
    "  python3 picchio.py /path/to/model.gguf\n"
    "  python3 picchio.py some-tag:latest")


def human_size(nbytes):
    """Model-file scale: GiB with one decimal, MiB below one GiB,
    blank when the source offered nothing."""
    if not nbytes:
        return ""
    if nbytes >= 1024 ** 3:
        return "{:.1f} GiB".format(nbytes / float(1024 ** 3))
    return "{:.0f} MiB".format(nbytes / float(1024 ** 2))


def _sourced(note, size):
    """'ollama, 5.3 GiB' for a human; just the source when size is blank."""
    return note + (", " + size if size else "")


def scan_models():
    """Look around this machine (read only, fast) for models it can run:
    ollama tags (the live api, or the manifest folder when ollama is not
    up), then .gguf files in this folder, the HF cache and the LM Studio
    folders. Returns (label, note, arg, size) rows: label, note and size
    name the source for a human (size stays blank when nothing cheap
    reports it), arg is the exact string the pipeline runs."""
    ollama = []
    if ollama_reachable():
        try:
            for m in ollama_api("/api/tags", timeout=5).get("models", []):
                if m.get("name"):
                    ollama.append((m["name"], "ollama", m["name"],
                                   human_size(m.get("size"))))
        except (urllib.error.URLError, OSError, ValueError):
            pass
    else:
        base = os.path.expanduser("~/.ollama/models/manifests")
        for reg in glob.glob(os.path.join(base, "*", "*", "*", "*")):
            parts = reg.split(os.sep)
            full = "{}:{}".format(parts[-2], parts[-1])
            ollama.append((full, "ollama, not running", full, ""))

    patterns = (
        "*.gguf",
        "~/.cache/huggingface/hub/models--*/snapshots/*/*.gguf",
        "~/.cache/lm-studio/models/*/*/*.gguf",
        "~/.lmstudio/models/*/*/*.gguf",
    )
    seen, ggufs = set(), []
    for pat in patterns:
        for f in sorted(glob.glob(os.path.expanduser(pat))):
            real = os.path.realpath(f)
            base = os.path.basename(f).lower()
            if real in seen or "mmproj" in base or f.endswith(".partial"):
                continue
            seen.add(real)
            try:
                size = human_size(os.path.getsize(real))
            except OSError:
                size = ""
            ggufs.append((os.path.basename(f), "gguf", f, size))
    # cap what the menu shows, but count the overflow so the presenters can
    # say "and N more" instead of dropping models silently (a long ollama
    # library used to hide every tag past the eighth with no hint at all)
    shown = ollama[:8] + ggufs[:8]
    dropped = max(0, len(ollama) - 8) + max(0, len(ggufs) - 8)
    return shown, dropped


def print_discovery(cands, dropped=0):
    """No terminal to ask at (a pipe or a redirect): print the commands
    that reproduce a run instead of a menu, each still pasteable as is.
    dropped counts models found past the cap, named so the list never
    hides one without a trace."""
    print("picchio: no model given. Runnable on this machine:\n")
    rows = [('"{}"'.format(arg) if " " in arg else arg,
             _sourced(note, size)) for label, note, arg, size in cands]
    w = min(max(len(q) for q, _ in rows), 48)
    for q, note in rows:
        print("  python3 picchio.py {:<{w}} ({})".format(q, note, w=w))
    if dropped:
        print("  ... and {} more not shown.".format(dropped))
    print("\nPick one, or point it at any other .gguf path or ollama tag.")


def _ask_line(prompt):
    """Read one line for the single direction question. EOF or ctrl-c
    returns None: declining to answer ends the flow, it does not crash it."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def resolve_direction(cands, interactive, ask, emit, dropped=0):
    """The whole zero-argument entry decision, in one place and pure of
    real IO so every path is testable. cands is what scan_models found;
    interactive says a terminal is on both ends; ask(prompt) returns the
    next typed line (or None at EOF); emit(line) shows one status line.
    dropped is how many models the scan found past the display cap, named
    in the menu so a long library never hides a model without a trace.
    Returns (action, model): ('run', arg) to diagnose arg, ('print', None)
    to fall back to pasteable commands, ('stop', None) for nothing to run.
    The flow asks at most once, and only at a real fork; one model needs no
    menu, and after the answer nothing else is asked."""
    if not interactive:
        return ("print", None) if cands else ("stop", None)
    if not cands:
        emit("No models found.")
        raw = (ask("Model (path or tag): ") or "").strip()
        return ("run", raw) if raw else ("stop", None)
    if len(cands) == 1:
        label, note, arg, size = cands[0]
        emit("1 model found.")
        emit("Selected: {} ({}).".format(label, _sourced(note, size)))
        return ("run", arg)
    emit("{} models found.".format(len(cands) + dropped))
    emit("")
    w = min(max(len(c[0]) for c in cands), 44)
    for i, (label, note, arg, size) in enumerate(cands, 1):
        if len(label) > w:
            label = label[:w - 14] + "..." + label[-11:]
        emit("  {:>2}) {:<{w}}  {:>9}   {}".format(i, label, size, note,
                                                   w=w))
    if dropped:
        emit("  ... and {} more not shown; type its tag or path to "
             "run any.".format(dropped))
    emit("")
    while True:
        line = ask("Model (number, path, or tag): ")
        if line is None or not line.strip():
            return ("stop", None)
        raw = line.strip()
        if raw.isdigit():
            k = int(raw)
            if 1 <= k <= len(cands):
                label, note, arg, size = cands[k - 1]
                emit("Selected: {} ({}).".format(label,
                                                 _sourced(note, size)))
                return ("run", arg)
            emit("No model {} in the list.".format(k))
            continue
        emit("Selected: {}.".format(raw))
        return ("run", raw)


def ollama_unload(tag):
    # Unload first so the cold pass pays the true load cost; ollama
    # keeps models resident for 5 minutes by default, and a cold number
    # measured against a resident model means nothing.
    try:
        ollama_api("/api/generate", {"model": tag, "keep_alive": 0},
                   timeout=60)
    except (urllib.error.URLError, OSError, ValueError):
        pass


# ---------------------------------------------- engine: llama-server (http)
#
# A model url instead of a path or tag means a llama-server someone
# already has running; picchio measures it over its own http api instead
# of launching anything. The api exposes no layer counts, no memory fit
# and no init log (checked against /props on b9430: nothing gpu shaped
# in it), so placement rests on the two witnesses that need no
# confession, the os meter and the prefill/decode signature. And the
# server owns its weights for as long as it lives: there is no unload
# call, so no cold pass exists, and the block says so instead of
# dressing a warm number as one.

def server_api(url, path, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url + path, data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def server_health(url):
    """GET /health: (True, None) when the server is up with its model
    loaded, else (False, reason), a still-loading server (it answers
    503 with an error body) told apart from nothing answering at all."""
    try:
        d = server_api(url, "/health", timeout=5)
        if d.get("status") == "ok":
            return True, None
        return False, ("the server at {} answered /health but is not "
                       "ready: {}".format(url, json.dumps(d)[:120]))
    except urllib.error.HTTPError:
        return False, ("the server at {} is still loading its model; "
                       "try again when /health says ok.".format(url))
    except (urllib.error.URLError, OSError, ValueError):
        return False, ("no llama-server answered at {}.\nStart one "
                       "(llama-server -m model.gguf) or check the "
                       "url.".format(url))


_PROPS_CACHE = {}


def server_props(url):
    """/props, fetched once per url. Fields used here, verified on
    b9430: model_path, model_alias, build_info, and the per request
    context under default_generation_settings.n_ctx."""
    if url not in _PROPS_CACHE:
        try:
            _PROPS_CACHE[url] = server_api(url, "/props", timeout=10)
        except (urllib.error.URLError, OSError, ValueError):
            _PROPS_CACHE[url] = {}
    return _PROPS_CACHE[url]


def server_ctx(url):
    """The context size a request to this server actually gets, or '?'
    when /props does not say; a question mark in the block beats a
    protocol default the server never promised."""
    try:
        return int(server_props(url)["default_generation_settings"]["n_ctx"])
    except (KeyError, TypeError, ValueError):
        return "?"


def url_is_local(url):
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def map_server(resp, wall_s):
    """response.timings from /completion, keys verified on b9430:
    prompt_n / prompt_ms cover the prompt tokens actually evaluated
    this pass, predicted_n / predicted_ms the generated ones, cache_n
    the prompt tokens reused from the kv cache instead of evaluated."""
    d = blank_pass()
    d["wall_s"] = wall_s
    t = resp.get("timings") or {}
    if t.get("prompt_ms") and t.get("prompt_n"):
        d["prompt_ms"] = float(t["prompt_ms"])
        d["prompt_tokens"] = int(t["prompt_n"])
    if t.get("predicted_ms") and t.get("predicted_n"):
        d["eval_ms"] = float(t["predicted_ms"])
        d["eval_tokens"] = int(t["predicted_n"])
    return finish_rates(d)


def run_server_pass(url, log_path=None, prompt=BENCH_PROMPT):
    # cache_prompt false, or the warm passes lie: this build reuses the
    # prompt kv across requests by default, and the second request then
    # evaluates 4 of 457 prompt tokens (measured here), which turns the
    # warm prefill rate into a per call overhead number, the short
    # prompt trap again. Forcing a full prefill every pass is the same
    # discipline as the keep_alive:0 unload in ollama mode, applied per
    # request; the wall clock is picchio's own, wrapped around the call.
    t0 = time.monotonic()
    resp = server_api(url, "/completion", {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "seed": 7,
        "ignore_eos": True,
        "cache_prompt": False,
    })
    wall_s = time.monotonic() - t0
    keep_log(log_path, json.dumps(resp, indent=1))
    t = resp.get("timings") or {}
    if t.get("cache_n"):
        sys.stderr.write("picchio: the server reused {} prompt tokens "
                         "from its cache despite cache_prompt false; "
                         "prefill is not a full read this pass.\n".format(
                             t["cache_n"]))
    if resp.get("truncated"):
        sys.stderr.write("picchio: the server truncated the prompt to "
                         "fit its context; rates are not comparable.\n")
    return map_server(resp, wall_s)


# ----------------------------------------------------------- telemetry (os)
#
# The engine's stderr is a confession; ioreg is the OS's own meter and
# does not care what the engine wrote. While the passes run, a thread
# polls the GPU accelerator entry a few times a second, so the verdict
# can cross check the claimed placement against what the silicon was
# seen doing: utilization over the compute windows, and the memory step
# the weights make when they actually land on the GPU.

TELE_HZ = 4.0  # one ioreg call costs 14-18 ms on the test machine; the
               # measured decode disturbance at 4 Hz is in README limits
TELE_PAD_S = 0.3  # decode ends about this long before the process does

RE_TELE = {
    "dev": re.compile(r'"Device Utilization %"=(\d+)'),
    "ren": re.compile(r'"Renderer Utilization %"=(\d+)'),
    "til": re.compile(r'"Tiler Utilization %"=(\d+)'),
    "mem": re.compile(r'"In use system memory"=(\d+)'),
}


def read_gpu_stats():
    """One ioreg sample, or None when there is nothing to read."""
    try:
        r = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    m = re.search(r'"PerformanceStatistics" = \{(.*)\}', r.stdout)
    if not m:
        return None
    s = {"t": time.monotonic()}
    for key, rx in RE_TELE.items():
        mm = rx.search(m.group(1))
        s[key] = int(mm.group(1)) if mm else None
    return s if s["dev"] is not None else None


class _IOReport:
    """GPU power without sudo: IOReport is the private framework that
    powermetrics itself reads, and its energy counters answer any
    process. Private means it can move between macOS versions, so every
    call is guarded; when anything is missing or NULL, power quietly
    stays off the os line and nothing else changes."""

    SCALE = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9}

    def __init__(self):
        p = ctypes.c_void_p
        self.cf = ctypes.CDLL("/System/Library/Frameworks/"
                              "CoreFoundation.framework/CoreFoundation")
        self.io = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        for lib, name, res, args in (
            (self.cf, "CFStringCreateWithCString", p,
             [p, ctypes.c_char_p, ctypes.c_uint32]),
            (self.cf, "CFStringGetCString", ctypes.c_bool,
             [p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
            (self.cf, "CFDictionaryGetValue", p, [p, p]),
            (self.cf, "CFArrayGetCount", ctypes.c_long, [p]),
            (self.cf, "CFArrayGetValueAtIndex", p, [p, ctypes.c_long]),
            (self.cf, "CFRelease", None, [p]),
            (self.io, "IOReportCopyChannelsInGroup", p,
             [p, p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]),
            (self.io, "IOReportCreateSubscription", p,
             [p, p, ctypes.POINTER(p), ctypes.c_uint64, p]),
            (self.io, "IOReportCreateSamples", p, [p, p, p]),
            (self.io, "IOReportCreateSamplesDelta", p, [p, p, p]),
            (self.io, "IOReportChannelGetChannelName", p, [p]),
            (self.io, "IOReportChannelGetUnitLabel", p, [p]),
            (self.io, "IOReportSimpleGetIntegerValue", ctypes.c_int64,
             [p, ctypes.POINTER(ctypes.c_int32)]),
        ):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        chans = self.io.IOReportCopyChannelsInGroup(
            self._cfstr("Energy Model"), None, 0, 0, 0)
        if not chans:
            raise OSError("no Energy Model channels")
        subbed = ctypes.c_void_p()
        self._sub = self.io.IOReportCreateSubscription(
            None, chans, ctypes.byref(subbed), 0, None)
        if not self._sub:
            raise OSError("IOReport subscription failed")
        self._subbed = subbed
        self._key = self._cfstr("IOReportChannels")
        self._prev = self.io.IOReportCreateSamples(self._sub, subbed, None)
        self._t_prev = time.monotonic()

    def _cfstr(self, s):
        return self.cf.CFStringCreateWithCString(None, s.encode(),
                                                 0x08000100)

    def _pystr(self, ref):
        buf = ctypes.create_string_buffer(128)
        if ref and self.cf.CFStringGetCString(ref, buf, 128, 0x08000100):
            return buf.value.decode()
        return None

    def watts(self):
        """Average GPU watts since the previous call, or None."""
        cur = self.io.IOReportCreateSamples(self._sub, self._subbed, None)
        t = time.monotonic()
        if not cur or t <= self._t_prev:
            return None
        delta = self.io.IOReportCreateSamplesDelta(self._prev, cur, None)
        w = None
        arr = self.cf.CFDictionaryGetValue(delta, self._key) \
            if delta else None
        for i in range(self.cf.CFArrayGetCount(arr) if arr else 0):
            ch = self.cf.CFArrayGetValueAtIndex(arr, i)
            name = self._pystr(self.io.IOReportChannelGetChannelName(ch))
            if name == "GPU Energy":
                unit = (self._pystr(
                    self.io.IOReportChannelGetUnitLabel(ch)) or "").strip()
                scale = self.SCALE.get(unit)
                if scale:
                    j = self.io.IOReportSimpleGetIntegerValue(ch, None)
                    w = j * scale / (t - self._t_prev)
                break
        self.cf.CFRelease(self._prev)
        if delta:
            self.cf.CFRelease(delta)
        self._prev, self._t_prev = cur, t
        return w


class _NVML:
    """The NVIDIA meter on Linux: libnvidia-ml.so.1 is the library
    nvidia-smi itself reads, resolvable wherever the driver is
    installed (verified on driver 550.54.14, all eight core symbols).
    utilization.gpu is the percent of the last internal sample period
    (between 1/6 s and 1 s depending on the product, per the NVML
    docs) during which any kernel ran, so 4 Hz polling repeats values;
    medians are judged, so repeats change nothing. Symbols are guarded
    like the IOReport private framework: anything missing quietly
    drops its field and never touches the judgment."""

    # thermal slowdown bits of the clocks throttle reasons mask
    # (sw thermal 0x20, hw thermal 0x40); power caps are normal
    # operation and do not count as throttling here
    THERMAL = 0x20 | 0x40

    class _Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class _Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit_v2 failed")
        self.hdl = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(
                0, ctypes.byref(self.hdl)) != 0:
            # index 0 only: multi gpu selection is a later milestone,
            # and the README limits say so
            raise OSError("no nvml device 0")

    def sample(self):
        """One sample in the sampler's own shape, or None."""
        u = self._Util()
        if not (hasattr(self.lib, "nvmlDeviceGetUtilizationRates")
                and self.lib.nvmlDeviceGetUtilizationRates(
                    self.hdl, ctypes.byref(u)) == 0):
            return None
        s = {"t": time.monotonic(), "dev": int(u.gpu), "mem": None}
        m = self._Mem()
        if hasattr(self.lib, "nvmlDeviceGetMemoryInfo") \
                and self.lib.nvmlDeviceGetMemoryInfo(
                    self.hdl, ctypes.byref(m)) == 0:
            s["mem"] = int(m.used)
        p = ctypes.c_uint()
        if hasattr(self.lib, "nvmlDeviceGetPowerUsage") \
                and self.lib.nvmlDeviceGetPowerUsage(
                    self.hdl, ctypes.byref(p)) == 0:
            s["gpu_w"] = p.value / 1000.0
        return s

    def device_name(self):
        if not hasattr(self.lib, "nvmlDeviceGetName"):
            return None
        buf = ctypes.create_string_buffer(96)
        if self.lib.nvmlDeviceGetName(self.hdl, buf, 96) != 0:
            return None
        name = buf.value.decode(errors="replace").strip()
        # display form: the NVIDIA prefix drops so the gpu line and the
        # footer hold the 66 column budget; the raw name is in the log
        return re.sub(r"^NVIDIA\s+", "", name) or None

    def throttled(self):
        # newer drivers renamed the symbol; try both, judge the same bits
        for sym in ("nvmlDeviceGetCurrentClocksThrottleReasons",
                    "nvmlDeviceGetCurrentClocksEventReasons"):
            if hasattr(self.lib, sym):
                r = ctypes.c_ulonglong()
                if getattr(self.lib, sym)(
                        self.hdl, ctypes.byref(r)) == 0:
                    return bool(r.value & self.THERMAL)
        return False


def thermal_raised():
    """True when macOS itself says the machine is under thermal
    pressure (pmset -g therm): a raised warning level or a CPU speed
    limit under 100. Presentation only; it never votes on placement."""
    out = _cmd_out(["pmset", "-g", "therm"])
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    if m and int(m.group(1)) < 100:
        return True
    m = re.search(r"thermal warning level\s*=?\s*(\d+)", out, re.I)
    return bool(m and int(m.group(1)) > 0)


def telemetry_start(disabled=False):
    """A running sampler, or a dict naming why there is none. The os
    line prints that reason, so a run without OS evidence says so
    instead of quietly reading like a fully instrumented one. macOS
    samples through ioreg plus IOReport; Linux through NVML when an
    NVIDIA driver is installed (amd/rocm is a separate milestone)."""
    if disabled:
        return {"off": "disabled"}
    sysname = platform.system()
    if sysname == "Linux":
        try:
            nv = _NVML()
            first = nv.sample()
        except Exception:
            return {"off": "no nvml"}
        if first is None:
            return {"off": "nvml gave no gpu stats"}
        return GpuSampler(first, backend=nv)
    if sysname != "Darwin":
        return {"off": "not macos"}
    if not shutil.which("ioreg"):
        return {"off": "no ioreg"}
    first = read_gpu_stats()
    if first is None:
        return {"off": "ioreg gave no gpu stats"}
    return GpuSampler(first)


class GpuSampler:
    def __init__(self, first, backend=None):
        self.samples = [first]
        self.marks = []
        # backend None is the macOS pair (ioreg samples, IOReport
        # watts); a backend object answers sample()/throttled() itself.
        # Only the sample source varies: the tick, the marks and the
        # window math below are the same physics on every platform.
        self._backend = backend
        self._power = None
        if backend is None:
            try:
                self._power = _IOReport()
            except Exception:
                self._power = None  # private API absent or moved: no watts
        self._hot = backend.throttled() if backend else thermal_raised()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        period = 1.0 / TELE_HZ
        while not self._stop.is_set():
            tick = time.monotonic()
            s = self._backend.sample() if self._backend \
                else read_gpu_stats()
            if s:
                if self._power:
                    try:
                        s["gpu_w"] = self._power.watts()
                    except Exception:
                        self._power = None
                self.samples.append(s)
            self._stop.wait(max(0.05, period - (time.monotonic() - tick)))

    def mark_pass(self, p):
        """Called the moment a pass returns: pins the pass to the wall
        clock, with the engine's own phase durations for the windows."""
        self.marks.append({
            "t_end": time.monotonic(), "wall_s": p["wall_s"],
            "load_s": (p["load_ms"] or 0) / 1000.0,
            "prompt_s": (p["prompt_ms"] or 0) / 1000.0,
            "eval_s": (p["eval_ms"] or 0) / 1000.0,
        })

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        hot = self._backend.throttled() if self._backend \
            else thermal_raised()
        return telemetry_summary(self.samples, self.marks,
                                 self._hot or hot,
                                 "nvml" if self._backend else "ioreg")


def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def telemetry_summary(samples, marks, hot=False, src=None):
    """Distills the timeline into what the verdict and the os line use:
    the idle baseline before pass 1, utilization over the compute
    windows, and the memory step inside the pass windows. Windows are
    tail aligned: decode ends at the pass end minus a small pad and
    prefill sits right before it; checked against the engine's own
    phase durations on the test machine (the gap between a head aligned
    load end and a tail aligned prefill start measured about 0.6 s)."""
    marks = [m for m in marks if m["wall_s"] and m["eval_s"]]
    if not samples or not marks:
        return {"off": "no samples"}
    t_first = marks[0]["t_end"] - marks[0]["wall_s"]
    idle = [s["dev"] for s in samples if s["t"] < t_first]
    work, work_w, mem_run = [], [], []
    for m in marks:
        dec1 = m["t_end"] - TELE_PAD_S
        pre0 = dec1 - m["eval_s"] - m["prompt_s"]
        t0 = m["t_end"] - m["wall_s"]
        for s in samples:
            if pre0 <= s["t"] <= dec1:
                work.append(s["dev"])
                work_w.append(s.get("gpu_w"))
            if t0 <= s["t"] <= m["t_end"] and s["mem"] is not None:
                mem_run.append(s["mem"])
    mem_base = _med([s["mem"] for s in samples if s["t"] < t_first])
    step = None
    if mem_base is not None and mem_run:
        step = max(0, max(mem_run) - mem_base)
    work = [w for w in work if w is not None]
    return {
        "hz": TELE_HZ, "n": len(samples), "src": src,
        "idle_med": _med(idle), "work_med": _med(work),
        "work_n": len(work), "mem_step": step,
        "work_w": _med(work_w), "throttled": bool(hot),
    }


def telemetry_vote(tele, rep, mode):
    """The OS evidence's vote on the engine's placement claim: agree,
    contradict or abstain. Only a full offload claim is judged; this
    tool hunts fake GPU claims, it does not overturn an engine that
    already confessed to CPU. Calibration on the test machine: a full
    Metal offload ran its compute windows at a median 99 device
    utilization with a +6.5 GiB memory step; a forced CPU run stayed
    at a median 0 with single-sample spikes to 53 from the desktop,
    which is why medians are judged and peaks are not."""
    if not tele or tele.get("off"):
        return "off"
    if mode == "ollama":
        full = rep["vram_frac"] is not None and rep["vram_frac"] >= 0.95
    else:
        n, total = rep["offload_n"], rep["offload_total"]
        full = n is not None and total and n >= total
    if not full:
        return "na"
    if tele["idle_med"] is None or tele["work_med"] is None \
            or tele["work_n"] < 6:
        return "abstain"
    if tele["idle_med"] > 25:
        # ioreg counts the whole GPU; on a busy desktop none of it can
        # be pinned on this one process, so the numbers stop judging
        return "abstain"
    if tele["work_med"] >= 50:
        return "agree"
    if tele["work_med"] < tele["idle_med"] + 15:
        mb = rep.get("model_bytes")
        if mb and tele["mem_step"] is not None \
                and tele["mem_step"] >= 0.5 * mb:
            return "abstain"  # the memory step says the weights landed
        return "contradict"
    return "abstain"


def telemetry_read(tele):
    """The OS meter's own reading when there is no engine claim to
    judge (a server endpoint): 'busy', 'flat', or None when it cannot
    testify. Same gates as the vote: enough samples over the compute
    windows and an idle machine before the run; a middling median stays
    silent rather than guessing."""
    if not tele or tele.get("off"):
        return None
    if tele["idle_med"] is None or tele["work_med"] is None \
            or tele["work_n"] < 6:
        return None
    if tele["idle_med"] > 25:
        return None
    if tele["work_med"] >= 50:
        return "busy"
    if tele["work_med"] < tele["idle_med"] + 15:
        return "flat"
    return None


def os_line(tele):
    """The one line of OS evidence in the block, None only when the
    render has no telemetry context at all (pre-telemetry replays)."""
    if tele is None:
        return None
    if tele.get("off"):
        return "gpu not sampled ({}); evidence: {}".format(
            tele["off"], tele.get("ev", "engine+timing"))
    if tele["idle_med"] is not None and tele["idle_med"] > 25:
        return "gpu {:.0f}% busy before the run, not idle; not judged" \
            .format(tele["idle_med"])
    parts = []
    if tele["idle_med"] is not None:
        parts.append("idle {:.0f}%".format(tele["idle_med"]))
    if tele["work_med"] is not None:
        parts.append("work {:.0f}%".format(tele["work_med"]))
    if tele["mem_step"] is not None:
        parts.append("mem +{:.1f} GiB".format(tele["mem_step"] / 1024 ** 3))
    w = tele.get("work_w")
    if w is not None:
        parts.append("{:.1f} W".format(w) if w < 100 else
                     "{:.0f} W".format(w))
    if tele.get("throttled"):
        parts.append("throttled")
    if not parts:
        return "gpu sampled, nothing usable came back"
    return "gpu " + ", ".join(parts)


# ------------------------------------------------------------- aggregation

def warm_stats(passes, key):
    vals = [p[key] for p in passes[1:] if p.get(key)]
    if not vals:
        return None, None, None
    return statistics.median(vals), min(vals), max(vals)


def build_rep(passes):
    """Evidence from the last pass, rates replaced by warm medians."""
    rep = dict(passes[-1])
    for key in ("prefill_toks", "decode_toks", "wallclock_toks"):
        med, _, _ = warm_stats(passes, key)
        rep[key] = med or rep.get(key)
    return rep


# ------------------------------------------------------- WHY attribution

def placement_flags(argv):
    """Placement flags found on the engine command line, verbatim."""
    names = ("-ngl", "--n-gpu-layers", "--gpu-layers", "--device", "-dev")
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        for n in names:
            if tok == n and i + 1 < len(argv):
                out.append((n, argv[i + 1]))
                i += 1
                break
            if tok.startswith(n + "="):
                out.append((n, tok.split("=", 1)[1]))
                break
        i += 1
    return out


def effective_ctx(extra):
    """The ctx the engine actually got: the protocol default unless the
    passthrough args override it (llama.cpp honors the last -c given;
    picchio's own -c comes first on the command line)."""
    ctx = CTX
    for i, tok in enumerate(extra):
        if tok.startswith(("-c=", "--ctx-size=")):
            tok, val = tok.split("=", 1)
        else:
            val = extra[i + 1] if i + 1 < len(extra) else ""
        if tok in ("-c", "--ctx-size") and val.isdigit():
            ctx = int(val)
    return ctx


def attribute_why(state, rep, mode, engine_argv):
    """One WHY line for a degraded verdict, None otherwise. Climbs a
    fixed evidence ladder and stops at the first rung with real evidence
    behind it: an explicit flag the user passed, the engine's own memory
    fit figures, a backend init failure line. Every rung requires its
    evidence to be present in this run; when none is, the honest answer
    is the word unknown, not a plausible guess."""
    if state not in ("SILENT CPU FALLBACK", "PARTIAL OFFLOAD"):
        # a conflict never takes a WHY line: the ladder attributes a
        # proven degradation, and a conflict is two sources disagreeing
        # about whether one happened at all. The paragraph names the
        # fight; that is the attribution.
        return None
    why = None
    if mode == "ollama":
        # the ollama api exposes no command line, no fit log and no
        # init log, so the ladder has no rungs to climb here
        why = "unknown: not in the ollama api (check the server log)"
    elif mode == "server":
        # same blindness over http: /props carries no placement fields
        # at all (checked on b9430), so the cause lives in the server's
        # own stderr, not in anything picchio can reach
        why = "unknown: not in the server api (check its stderr log)"
    else:
        n, total = rep["offload_n"], rep["offload_total"]
        if n is None:
            # a fallback verdict with no engine claim at all is the
            # silent-engine conviction: the cause this run can prove is
            # the physics itself, and the WHY states it without
            # guessing at the build (a cpu only build can be deliberate)
            why = "no gpu evidence in the log; the gpu meter stayed idle"
            why = "WHY: " + why
            return why
        forced = []
        for name, val in placement_flags(engine_argv):
            # a flag only counts as the cause when its value matches the
            # placement the engine delivered; a flag that asked for more
            # GPU than was given did not cause the shortfall
            if (name in ("--device", "-dev")
                    and val.lower() in ("none", "cpu") and n == 0):
                forced.append("{} {}".format(name, val))
            elif (name in ("-ngl", "--n-gpu-layers", "--gpu-layers")
                    and val.isdigit() and n is not None
                    and int(val) == n and total and n < total):
                forced.append("{} {}".format(name, val))
        if forced:
            why = "forced by flag: " + " ".join(forced)
        elif (rep["fit_seen"] and rep["free_mib"] is not None
                and n is not None and total and n < total):
            why = "memory fit: saw {} MiB free, gave {}/{} layers".format(
                rep["free_mib"], n, total)
        elif rep["init_fail"]:
            why = rep["init_fail"]
        else:
            why = "unknown: the engine log does not say why"
    why = "WHY: " + why
    if len(why) > WIDTH:
        why = why[:WIDTH - 3] + "..."
    return why


# --------------------------------------------------------------- diagnosis

def diagnose(cold, rep, mode, tele=None):
    """Returns (state, paragraph). State drives the exit code.

    Three evidence sources vote: the engine's own confession (offload
    lines, ollama ps), the OS meter (ioreg utilization and memory, when
    sampled), and timing physics (the prefill/decode signature ratio).
    A full offload claim earns HEALTHY only while no source actively
    contradicts it; any two sources fighting is CONFLICTING EVIDENCE
    with the fight spelled out. A missing source abstains and the os
    line says what was missing; it never quietly counts as agreement.

    The block must stay inside 15 lines; the renderer drops trailing
    sentences from the paragraph until it fits, so the load bearing
    sentence goes first."""
    decode = rep["decode_toks"] or cold["decode_toks"]
    prefill = rep["prefill_toks"] or cold["prefill_toks"]
    wait_s = 2500.0 / prefill if prefill else None
    vote = telemetry_vote(tele, rep, mode)

    def fallback_para():
        # prefill leads: with an os line and a WHY line in the block the
        # budget leaves this paragraph one line, and the hidden cost is
        # the sentence that must survive (decode's alibi shows in the
        # table right above)
        bits = []
        if prefill:
            bits.append("Prefill: {:.0f} s per 2500 "
                        "tokens.".format(wait_s))
        if decode:
            bits.append("Decode ({:.1f}) looks passable; that is how "
                        "this hides.".format(decode))
        return " ".join(bits) or "The gpu line above is the story."

    if mode == "server":
        # no confession exists over http: the server api exposes neither
        # layer counts nor a memory split, so the two witnesses that
        # need none vote on their own. Cutoffs are the calibrated ones
        # the other modes already use: work median 50 for busy, ratio
        # under 5 cpu shaped, 15 and over gpu shaped (healthy gpu runs
        # measured 20-44x here, cpu runs 2.3-5x).
        votes = []
        osr = telemetry_read(tele)
        if osr == "busy":
            votes.append(("the os meter saw the gpu work at "
                          "{:.0f}%".format(tele["work_med"]), "gpu"))
        elif osr == "flat":
            votes.append(("the os meter saw the gpu stay flat while the "
                          "tokens were made", "cpu"))
        ratio = prefill / decode if prefill and decode else None
        if ratio is not None and ratio >= 15:
            votes.append(("prefill ran {:.0f}x decode, gpu "
                          "shaped".format(ratio), "gpu"))
        elif ratio is not None and ratio < 5:
            votes.append(("prefill at {:.1f}x decode is cpu "
                          "shaped".format(ratio), "cpu"))
        shapes = {s for _, s in votes}
        if len(shapes) == 2:
            para = "{}; {}. No engine claim breaks the tie. Believe " \
                   "neither.".format(votes[0][0], votes[1][0])
            return "CONFLICTING EVIDENCE", para[0].upper() + para[1:]
        if shapes == {"cpu"}:
            para = " and ".join(t for t, _ in votes) \
                + ": the tokens were made on the cpu."
            if wait_s:
                para += " Prefill: {:.0f} s per 2500 tokens.".format(wait_s)
            return "SILENT CPU FALLBACK", para[0].upper() + para[1:]
        if shapes == {"gpu"}:
            para = " and ".join(t for t, _ in votes) \
                + ": the gpu did the work."
            if decode:
                para += (" Quote the warm median decode: {:.1f} "
                         "tok/s.".format(decode))
            return "HEALTHY", para[0].upper() + para[1:]
        return "NO PLACEMENT EVIDENCE", (
            "The server api exposes no placement, and neither the os "
            "meter nor the timing signature was decisive here. Rates "
            "are measured; placement is not."
        )

    if mode == "ollama":
        frac = rep["vram_frac"]
        if frac is None:
            return "NO PLACEMENT EVIDENCE", (
                "Ollama did not report a memory split for this model, so "
                "picchio cannot say where it ran. Rates are measured; "
                "placement is not."
            )
        if frac < 0.05:
            return "SILENT CPU FALLBACK", fallback_para()
        if frac < 0.95:
            return "PARTIAL OFFLOAD", (
                "{:.0f}% of weights sat on CPU; expect rates below a "
                "fully offloaded run.".format(100 - frac * 100)
            )
        # ollama's reported split has been known to disagree with where
        # the kernels actually ran, so a full-GPU claim is cross checked
        # against the OS meter and the speed signature before HEALTHY.
        if vote == "contradict":
            return "CONFLICTING EVIDENCE", (
                "Ollama says 100% GPU; the OS saw the GPU stay flat "
                "while the tokens were made. Believe neither."
            )
        if prefill and decode and prefill < 5 * decode:
            return "CONFLICTING EVIDENCE", (
                "Ollama says 100% GPU; prefill at only {:.1f}x decode "
                "is CPU shaped. Believe neither.".format(prefill / decode)
            )
        para = "Ollama reports 100% of weights in GPU memory."
        if decode:
            para += (" Quote the warm median decode: {:.1f} "
                     "tok/s.".format(decode))
        if prefill and decode and prefill > 3 * decode:
            para += (" {:.0f} tok/s is prefill: reading, not "
                     "writing.".format(prefill))
        return "HEALTHY", para

    n, total = rep["offload_n"], rep["offload_total"]
    if n is None:
        # silent-engine conviction, the linux killer case: a build with
        # no gpu support prints no placement evidence anywhere, yet the
        # machine has an nvidia gpu the os meter can see. Five gates,
        # every one required: no engine evidence at all (no offload
        # line and no device line), the nvml meter present, an idle
        # baseline with a flat compute window (telemetry_read's own
        # gates), and no exculpatory memory step (weights that landed
        # on the gpu veto the conviction, same veto the vote uses).
        # The prefill/decode ratio is deliberately not a gate: the
        # misbuilt 4090 fixture measured prefill at 15.1x decode on 48
        # EPYC threads, so the laptop calibrated 5x line does not
        # transfer to many core machines.
        if tele and tele.get("src") == "nvml" and not rep["gpu_kind"] \
                and telemetry_read(tele) == "flat":
            mb = rep.get("model_bytes")
            stepped = mb and tele.get("mem_step") is not None \
                and tele["mem_step"] >= 0.5 * mb
            if not stepped:
                # cost sentence first: the renderer drops sentences from
                # the end under the 15 line budget, the WHY line already
                # carries the evidence, and the 89 char evidence sentence
                # cannot survive a one line squeeze (the 4090 retest cut
                # it mid word as "eviden..")
                para = ("This build printed no gpu evidence and the "
                        "gpu stayed idle while the tokens were made.")
                if wait_s:
                    para = "Prefill: {:.0f} s per 2500 tokens. ".format(
                        wait_s) + para
                return "SILENT CPU FALLBACK", para
        return "NO PLACEMENT EVIDENCE", (
            "This build did not report layer placement, so picchio cannot "
            "prove where the model ran. Rates are measured; placement is "
            "not. A newer llama.cpp build logs it."
        )
    if n == 0:
        return "SILENT CPU FALLBACK", fallback_para()
    if total and n < total:
        return "PARTIAL OFFLOAD", (
            "{} layers sat on CPU; expect rates below a fully "
            "offloaded run.".format(total - n)
        )
    # a full offload claim from stderr, cross checked the same way the
    # ollama one is: first the OS meter, then the speed signature
    if vote == "contradict":
        return "CONFLICTING EVIDENCE", (
            "The engine says {}/{} layers on GPU; the OS saw the GPU "
            "stay flat while the tokens were made. Believe "
            "neither.".format(n, total)
        )
    if prefill and decode and prefill < 5 * decode:
        return "CONFLICTING EVIDENCE", (
            "The engine says {}/{} layers on GPU; prefill at only "
            "{:.1f}x decode is CPU shaped. Believe neither.".format(
                n, total, prefill / decode)
        )
    para = "The GPU did the work."
    if decode:
        para += (" Quote the warm median decode: {:.1f} tok/s.".format(
            decode))
    if prefill and decode and prefill > 3 * decode:
        para += (" {:.0f} tok/s is prefill: reading speed, not "
                 "writing.".format(prefill))
    return "HEALTHY", para


def classify_number(x, rates):
    """rates: dict lane -> tok/s (may contain None). Returns (verdict, para)."""
    if x <= 0:
        return "NOT A RATE", "tok/s numbers are positive; nothing to check."
    lanes = [(k, v) for k, v in rates.items() if v]
    if not lanes:
        return "NOTHING TO COMPARE AGAINST", "No measured rates available."
    best, best_ratio = None, None
    for k, v in lanes:
        ratio = x / v
        off = max(ratio, 1 / ratio)
        if best_ratio is None or off < best_ratio:
            best, best_ratio = k, off
    lane_desc = {
        "prefill": "prompt reading speed, not generation speed",
        "decode": "generation speed, the number worth comparing",
        "wallclock": "tokens over total wall time, load and all",
    }
    measured = ", ".join("{} {:.1f}".format(k, v) for k, v in lanes)
    # The 1.30 band: wide enough to absorb the drift measured here
    # (same weights across two runtimes differed 12% on decode; warm
    # passes repeat within a few percent), narrow enough that decode
    # and wallclock, 1.4x apart on this machine, cannot both claim the
    # same number.
    if best_ratio <= 1.30:
        para = ("{:.1f} tok/s sits within {:.0f}% of the {} rate measured "
                "here. That reads like {}. (measured: {} tok/s)".format(
                    x, (best_ratio - 1) * 100, best, lane_desc[best],
                    measured))
        return "READS LIKE " + best.upper(), para
    para = ("{:.1f} tok/s is not within 30% of anything measured here "
            "(closest: {}, off by {:.1f}x; measured: {} tok/s). Before "
            "trusting that number, ask which of the three rates it was, "
            "and on what hardware, quant, and context length.".format(
                x, best, best_ratio, measured))
    return "MATCHES NOTHING MEASURED HERE", para


# --------------------------------------------------------------- rendering

def fmt_rate(v):
    return "{:.1f} tok/s".format(v) if v else "n/a"


def fmt_span(lo, hi, big=False):
    if lo is None:
        return "-"
    f = "{:.0f}~{:.0f}" if big else "{:.1f}~{:.1f}"
    return f.format(lo, hi)


def bar_line(label, secs, frac):
    barw = 28
    fill = max(0, min(barw, int(round(frac * barw))))
    return "  {:<13}{:>6.1f} s  {}{}  {:>3.0f}%".format(
        label, secs, "#" * fill, "." * (barw - fill), frac * 100
    )


def wrap_para(text):
    return textwrap.wrap(text, width=WIDTH - 2,
                         initial_indent="  ", subsequent_indent="  ")


def colorize(text, stream=None):
    """ANSI color for terminals only (stream defaults to stdout; guard
    passes stderr). Piped or redirected output stays pure ASCII, so a
    pasted block is identical to what the parser and the selftest see.
    NO_COLOR is respected. The id card is left unpainted on purpose:
    it is the paste totem, and it must look the same everywhere."""
    if os.environ.get("NO_COLOR") or not (stream or sys.stdout).isatty():
        return text
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"
    states = (("SILENT CPU FALLBACK", RED), ("CONFLICTING EVIDENCE", YELLOW),
              ("PARTIAL OFFLOAD", YELLOW), ("NO PLACEMENT EVIDENCE", YELLOW),
              ("HEALTHY", GREEN), ("PASS", GREEN), ("FLAG", RED))
    out = []
    for line in text.splitlines():
        if line.startswith("VERDICT: "):
            for state, col in states:
                if state in line:
                    line = line.replace(state, BOLD + col + state + RESET, 1)
                    break
        elif line.startswith("gpu "):
            for word, col in (("NOT ENGAGED", RED), ("EVIDENCE UNKNOWN",
                              YELLOW), ("NO EVIDENCE", YELLOW),
                              ("PARTIAL", YELLOW), ("ENGAGED", GREEN)):
                if word in line:
                    line = line.replace(word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith("GPU "):
            for word, col in (("GPU BUSY", GREEN), ("GPU IDLE", RED),
                              ("GPU MIXED", YELLOW),
                              ("GPU UNREADABLE", YELLOW)):
                if line.startswith(word):
                    line = line.replace(word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith("picchio guard: "):
            for word, col in (("NOT ENGAGED", RED),
                              ("SILENT CPU FALLBACK", RED),
                              ("PARTIAL OFFLOAD", YELLOW),
                              ("ENGAGED", GREEN)):
                if word in line:
                    line = line.replace(word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith("picchio monitor: "):
            for word, col in (("NOT ENGAGED", RED),
                              ("SILENT CPU FALLBACK", RED),
                              ("UNSURE", YELLOW),
                              ("ENGAGED", GREEN)):
                if word in line:
                    line = line.replace(word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith("SUSPECT: "):
            line = BOLD + YELLOW + "SUSPECT" + RESET + line[7:]
        elif line.startswith("  verdict"):
            for word, col in (("not judged", None), ("fits", GREEN),
                              ("tight", YELLOW), ("no", RED)):
                if word in line:
                    if col:
                        line = line.replace(
                            word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith(("WHY: ", "-- picchio")) or (
                line.startswith(("ctx ", "depth"))
                and "prefill" in line and "wallclock" in line
                and "tok/s" not in line):
            line = DIM + line + RESET
        elif line.startswith(("YOUR NUMBER: ", "SLOPE: ")):
            line = BOLD + line + RESET
        out.append(line)
    return "\n".join(out)


def menu_paint(line):
    """Discovery-menu color under colorize's contract: terminals only,
    NO_COLOR respected, and the plain text underneath is exactly what
    the selftest asserts on. Names carry the eye; the size and source
    columns sit dim behind them."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return line
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    if re.match(r"^\d+ models? found\.$", line):
        return BOLD + line + RESET
    m = re.match(r"^(  +\d+\) )(.*?)(  +\S.*)$", line)
    if m:
        return m.group(1) + m.group(2) + DIM + m.group(3) + RESET
    return line


def gpu_line(rep, mode):
    if mode == "server":
        return "NO EVIDENCE (the server api exposes no placement)"
    if mode == "ollama":
        frac = rep["vram_frac"]
        if frac is None:
            return "EVIDENCE UNKNOWN (ollama gave no memory split)"
        pct = "{:.0f}% of weights in GPU memory (ollama ps)".format(
            frac * 100)
        if frac < 0.05:
            return "NOT ENGAGED: " + pct
        if frac < 0.95:
            return "PARTIAL: " + pct
        return "ENGAGED: " + pct
    n, total = rep["offload_n"], rep["offload_total"]
    if n is None:
        return "NO EVIDENCE (engine did not report layer placement)"
    if n == 0:
        g = "NOT ENGAGED: 0/{} layers on GPU".format(total)
    elif n < total:
        g = "PARTIAL: {}/{} layers on GPU".format(n, total)
    else:
        g = "ENGAGED: {}/{} layers on GPU".format(n, total)
    if rep["gpu_kind"] and rep["gpu_device"]:
        g += " ({}: {})".format(rep["gpu_kind"], rep["gpu_device"])
    elif rep["gpu_kind"]:
        g += " ({})".format(rep["gpu_kind"])
    return g


def render_verdict(mach, engine_str, model_name, passes, state, para, mode,
                   explain_part=None, cold_note=None, why=None,
                   ctx=CTX, extra=(), tele=None):
    """The block stays inside 15 lines, kept narrow so it survives
    pasting into a forum comment (a long model name can push line one
    wider). The budget is a feature; never add lines without removing."""
    cold = passes[0]
    rep = build_rep(passes)
    out = []
    bits = [model_name]
    if rep.get("model_params"):
        bits.append(rep["model_params"])
    if rep.get("model_size"):
        bits.append(rep["model_size"])
    bits.append(engine_str)
    out.append("model    " + ", ".join(bits))
    gline = "gpu      " + gpu_line(rep, mode)
    if extra:
        # passthrough args (the -ngl asked for, sampling overrides) ride
        # the gpu line: asked-for belongs next to delivered, and a new
        # line would break the budget. Truncated if long, never dropped:
        # on a new-format block, no bracket must mean no extra args.
        astr = " ".join(extra)
        room = WIDTH - len(gline) - 3
        if len(astr) > room:
            astr = astr[:max(2, room) - 2] + ".."
        gline += " [" + astr + "]"
    out.append(gline)
    oline = os_line(tele)
    if oline:
        # the OS's independent reading, right under the engine's claim;
        # absent only on replays of runs that predate the sampler
        out.append("os       " + oline)
    # ctx rides the dead gutter before the lane headers: the only
    # always-blank columns in the block ("ctx 9999999" just fits 11)
    out.append("{:<11}{:>13}  {:>13}  {:>13}".format(
        "ctx " + str(ctx), "prefill", "decode", "wallclock"))
    if mode != "server":
        out.append("  {:<9}{:>13}  {:>13}  {:>13}".format(
            "cold", fmt_rate(cold["prefill_toks"]),
            fmt_rate(cold["decode_toks"]),
            fmt_rate(cold["wallclock_toks"])))
    pm, plo, phi = warm_stats(passes, "prefill_toks")
    dm, dlo, dhi = warm_stats(passes, "decode_toks")
    wm, wlo, whi = warm_stats(passes, "wallclock_toks")
    out.append("  {:<9}{:>13}  {:>13}  {:>13}".format(
        "warm mid", fmt_rate(pm), fmt_rate(dm), fmt_rate(wm)))
    out.append("  {:<9}{:>13}  {:>13}  {:>13}".format(
        "warm span", fmt_span(plo, phi, big=True), fmt_span(dlo, dhi),
        fmt_span(wlo, whi)))

    if mode == "server":
        # the server owned the weights before pass 1, so no cold pass
        # exists: no cold row, no load bar, and one line saying so
        # beats a warm number dressed up as a cold one
        out.append("cold start not measured: the server already owned "
                   "the weights")
    else:
        wall = cold["wall_s"] or 0
        load_s = (cold["load_ms"] or 0) / 1000.0
        prefill_s = (cold["prompt_ms"] or 0) / 1000.0
        decode_s = (cold["eval_ms"] or 0) / 1000.0
        other_s = max(0.0, wall - load_s - prefill_s - decode_s)
        title = "where the cold pass went ({:.1f} s".format(wall)
        if rep.get("threads"):
            title += ", {}/{} threads".format(rep["threads"], rep["cores"])
        if cold_note:
            title += ", weights cached"
        out.append(title + ")")
        if wall > 0:
            out.append(bar_line("load weights", load_s, load_s / wall))
            out.append(bar_line("prefill", prefill_s, prefill_s / wall))
            out.append(bar_line("decode", decode_s, decode_s / wall))
            out.append(bar_line("engine misc", other_s, other_s / wall))
    # the 15 line budget is enforced, not hoped for: however many
    # optional lines rode in (the os line, a WHY line), trailing
    # sentences drop from the paragraph until the block fits
    fixed = len(out) + (1 if why else 0) + 1  # + WHY + footer
    vlines = textwrap.wrap("VERDICT: {}. {}".format(state, para),
                           width=WIDTH - 2, subsequent_indent="  ")
    while len(vlines) > max(1, 15 - fixed):
        body = para.rstrip()[:-1]
        cut = body.rfind(". ")  # whole sentences drop first,
        if cut >= 0:
            para = para[:cut + 1]
        else:
            cut = body.rfind("; ")  # then a trailing clause
            if cut < 0:
                break
            para = para[:cut] + "."
        vlines = textwrap.wrap("VERDICT: {}. {}".format(state, para),
                               width=WIDTH - 2, subsequent_indent="  ")
    room = max(1, 15 - fixed)
    if len(vlines) > room:
        # a single uncuttable sentence can still overflow; the budget
        # is enforced, not hoped for, so truncate as the last resort
        vlines = vlines[:room]
        vlines[-1] = vlines[-1][:WIDTH - 4].rstrip() + ".."
    out.extend(vlines)
    if why:
        out.append(why)
    if explain_part:
        out.append("YOUR NUMBER: " + explain_part[0])
        out.extend(wrap_para(explain_part[1]))
    out.append("-- picchio v{} {} on {}, {} GB, {}".format(
        VERSION, PROTOCOL, mach["chip"], mach["ram_gb"] or "?", mach["os"]))
    return "\n".join(out)


# ------------------------------------------------------------------- guard

RE_GUARD_OFF = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")
# lines worth pinning for the exit summary even after the tail window
# has rolled past them: placement, fit, device and init evidence
RE_GUARD_PIN = re.compile(
    r"offloaded\s+\d+/\d+\s+layers|model buffer size|MiB free|"
    r"common_params_fit_impl|ggml_metal|ggml_cuda|ggml_vulkan|"
    r"ggml_backend|system_info")
# a line that means the engine moved past loading; placement evidence
# seen by now is final even on builds that print no buffer lines
RE_GUARD_PAST = re.compile(
    r"system_info|prompt eval time|listening|server is listening|"
    r"main: server", re.I)


def guard_why(rep, cmd):
    """WHY attribution for a degraded placement seen by guard; None when
    the placement is full (a healthy load needs no cause assigned)."""
    n, total = rep["offload_n"], rep["offload_total"]
    if n is None or not total or n >= total:
        return None
    state = "SILENT CPU FALLBACK" if n == 0 else "PARTIAL OFFLOAD"
    return attribute_why(state, rep, "llama.cpp", cmd)


def guard_state_line(rep, why):
    line = "picchio guard: " + gpu_line(rep, "llama.cpp")
    if why:
        line += "; " + why
    return line


def guard(cmd, keep_dir=None):
    """Wraps the user's own llama.cpp command (llama-server, llama-cli,
    anything that logs to stderr), tees its stderr through untouched,
    and speaks exactly twice on top of it: one placement line the moment
    the evidence is complete, and a short summary when the child exits.
    It never kills or signals the child: the requirement this mode comes
    from is a tool that warns but refuses to get in the way."""
    try:
        child = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True,
                                 errors="replace")
    except OSError as e:
        sys.exit("picchio guard: could not start {}: {}".format(cmd[0], e))
    t0 = time.monotonic()
    log = open(os.path.join(keep_dir, "guard.stderr.txt"), "w") \
        if keep_dir else None
    # pinned keeps the load time placement evidence forever; tail keeps
    # the recent perf lines. A guarded server can log for hours, so the
    # full stream is never held in memory (the placement evidence sits
    # well under the caps: 238 lines at -lv 4, 1.7k at -lv 5 here).
    pinned, tail = [], []
    announced = False
    pending = False  # an "offloaded n/total" line arrived, unconfirmed
    try:
        for line in child.stderr:
            sys.stderr.write(line)
            if log:
                log.write(line)
            stripped = line.rstrip("\n")
            tail.append(stripped)
            if len(tail) > 4000:
                del tail[:2000]
            if len(pinned) < 800 and RE_GUARD_PIN.search(stripped):
                pinned.append(stripped)
            if announced:
                continue
            if RE_GUARD_OFF.search(stripped):
                pending = True
                continue
            # the fit planning pass also prints an "offloaded" line, but
            # only the real load allocates buffers (its planning twin
            # reports 0.00 MiB and no _Mapped suffix), so an offloaded
            # line is confirmed by the next _Mapped buffer line, or by
            # any line that shows the engine already running
            if pending and ("_Mapped model buffer size" in stripped
                            or RE_GUARD_PAST.search(stripped)):
                rep = parse_stderr("\n".join(pinned), None)
                sys.stderr.write(colorize(
                    guard_state_line(rep, guard_why(rep, cmd)),
                    sys.stderr) + "\n")
                announced = True
    except KeyboardInterrupt:
        pass  # ctrl-c went to the child too; fall through to its exit
    try:
        code = child.wait()
    except KeyboardInterrupt:
        sys.exit(130)  # second ctrl-c: leave, still without killing it
    wall = time.monotonic() - t0
    if log:
        log.close()
    rep = parse_stderr("\n".join(pinned + tail[-1200:]), wall)
    out = ["picchio guard: {} exited {} after {:.1f} s".format(
        os.path.basename(cmd[0]), code, wall)]
    if rep["offload_n"] is None:
        out.append("picchio guard: no placement evidence appeared on "
                   "stderr; on llama.cpp builds where the default "
                   "verbosity hides it, add --verbose or -lv 4")
    else:
        out.append(guard_state_line(rep, guard_why(rep, cmd)))
    if rep["prefill_toks"] or rep["decode_toks"]:
        out.append("picchio guard: last rates seen: prefill {}, "
                   "decode {}".format(fmt_rate(rep["prefill_toks"]),
                                      fmt_rate(rep["decode_toks"])))
    sys.stderr.write(colorize("\n".join(out), sys.stderr) + "\n")
    # exit code: the child's own, passed through (128+N for a signal,
    # the shell convention). Measure mode owns its subprocess, so there
    # picchio's 0/2/3/4/5 codes are the product; here the subprocess is
    # the user's product, and scripts wrapping their server must keep
    # seeing the exit semantics they already depend on. The warning
    # lives on stderr, not in the code.
    sys.exit(code if code >= 0 else 128 - code)


def guard_cli(argv):
    keep = None
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio.py guard [--keep-logs DIR] -- <command...>\n"
              "wrap a llama.cpp command; warn on stderr the moment its\n"
              "own log shows layers landing off the GPU, never kill it,\n"
              "and print a placement summary when it exits.")
        sys.exit(0)
    if argv[:1] == ["--keep-logs"] and len(argv) > 1:
        keep = argv[1]
        os.makedirs(keep, exist_ok=True)
        argv = argv[2:]
    if argv[:1] != ["--"] or len(argv) < 2:
        sys.exit("picchio guard: usage: picchio.py guard "
                 "[--keep-logs DIR] -- <command...>")
    guard(argv[1:], keep)


# ----------------------------------------------------------------- compare

RE_QUANT = re.compile(r"\b(I?Q\d+(?:_[A-Z0-9]+)+|F16|BF16|F32)\b", re.I)


def parse_block(text):
    """Reads a pasted verdict block back into its variables. The input
    is a forum comment, so junk around the block is ignored. Fields the
    block does not carry stay None and print as unknown, never guessed;
    blocks from before the fingerprint fields have no ctx line, which
    is also how the two formats are told apart."""
    b = {k: None for k in ("model", "quant", "engine", "place", "frac",
                           "args", "ctx", "threads", "chip", "ram", "os",
                           "place_word", "verdict", "os_raw", "os_work",
                           "os_idle", "os_mem", "os_watts", "os_note")}
    rates = {}
    for line in text.splitlines():
        line = line.rstrip()
        m = re.match(r"model\s{4}(\S.*)", line)
        if m and b["model"] is None:
            b["model"] = m.group(1).split(",")[0].strip()
            em = re.search(r"((?:llama\.cpp|ollama)\s+\S+)$", m.group(1))
            b["engine"] = em.group(1) if em else None
            qm = RE_QUANT.search(m.group(1))
            b["quant"] = qm.group(1).upper() if qm else None
        m = re.match(r"gpu\s{6}(\S.*)", line)
        if m and b["place"] is None:
            g = m.group(1)
            am = re.search(r" \[(.+)\]$", g)
            if am:
                b["args"], g = am.group(1), g[:am.start()]
            for w in ("NOT ENGAGED", "PARTIAL", "ENGAGED",
                      "NO EVIDENCE", "EVIDENCE UNKNOWN"):
                if g.startswith(w):  # NOT ENGAGED before ENGAGED (substring)
                    b["place_word"] = w
                    break
            lm = re.search(r"(\d+)/(\d+) layers", g)
            pm = re.search(r"(\d+)% of weights", g)
            if lm and int(lm.group(2)):
                b["frac"] = int(lm.group(1)) / int(lm.group(2))
                b["place"] = "{}/{} layers on GPU".format(*lm.groups())
            elif pm:
                b["frac"] = int(pm.group(1)) / 100.0
                b["place"] = "{}% of weights on GPU".format(pm.group(1))
            else:
                b["place"] = g.split("(")[0].strip()
        m = re.match(r"ctx (\d+)\s+prefill", line)
        if m:
            b["ctx"] = int(m.group(1))
        m = re.match(r"\s{2}(cold|warm mid)\s{2,}(\S.*)", line)
        if m and m.group(1) not in rates:
            cells = re.findall(r"([\d.]+) tok/s|n/a", m.group(2))
            if len(cells) == 3:
                rates[m.group(1)] = [float(c) if c else None for c in cells]
        if line.startswith("where the cold pass went"):
            tm = re.search(r"(\d+/\d+) threads", line)
            if tm:
                b["threads"] = tm.group(1)
        m = re.match(r"os\s{2,}(gpu .*)", line)
        if m and b["os_raw"] is None:
            g = b["os_raw"] = m.group(1)
            if "not judged" in g:
                b["os_note"] = "not judged"
            elif "not sampled" in g:
                b["os_note"] = "not sampled"
            elif "nothing usable" in g:
                b["os_note"] = "unusable"
            for key, rx in (("os_idle", r"idle (\d+)%"),
                            ("os_work", r"work (\d+)%")):
                mm = re.search(rx, g)
                if mm:
                    b[key] = int(mm.group(1))
            mm = re.search(r"mem \+([\d.]+) GiB", g)
            if mm:
                b["os_mem"] = float(mm.group(1))
            mm = re.search(r"([\d.]+) W\b", g)
            if mm:
                b["os_watts"] = float(mm.group(1))
        m = re.match(r"VERDICT: (\S.*)", line)
        if m and b["verdict"] is None:
            for st in ("SILENT CPU FALLBACK", "PARTIAL OFFLOAD",
                       "NO PLACEMENT EVIDENCE", "CONFLICTING EVIDENCE",
                       "HEALTHY"):
                if m.group(1).startswith(st):
                    b["verdict"] = st
                    break
        m = re.match(r"-- picchio v\S+ \S+ on (.+), (\d+|\?) GB, (.+)", line)
        if m:
            b["chip"], b["ram"], b["os"] = m.groups()
    b["row"] = "warm mid" if "warm mid" in rates else \
        ("cold" if "cold" in rates else None)
    b["rates"] = rates.get(b["row"]) or [None] * 3
    return b if b["model"] and b["row"] else None


def base_model(b):
    """Model name normalized for identity: quant token, .gguf suffix and
    separators dropped, so Qwen3.5-9B-Q4_K_M.gguf and qwen3.5:9b read as
    the same weights. Registry tags also drop suffixes like -Instruct,
    so containment counts as a match; that rule is mechanical, not fuzzy."""
    s = re.sub(r"\.gguf$", "", b["model"], flags=re.I)
    return re.sub(r"[^a-z0-9]", "", RE_QUANT.sub("", s).lower())


def suspect_para(a, b):
    """The attribution ladder, mechanical and in fixed order: placement,
    then quantization, then a ctx an order of magnitude apart, then
    hardware. The first rung whose evidence differs takes the blame and
    the climb stops; a rung missing its evidence on either side is
    skipped and named, never guessed across. Returns (text, skipped)."""
    skipped = []

    def known(key):
        if a[key] is not None and b[key] is not None:
            return True
        skipped.append({"frac": "placement"}.get(key, key))
        return False

    ma, mb = base_model(a), base_model(b)
    if not (ma == mb or ma in mb or mb in ma):
        text = ("NOT COMPARABLE: different models ({} vs {}). The ladder "
                "ranks configuration, not models.".format(a["model"],
                                                          b["model"]))
    elif known("frac") and abs(a["frac"] - b["frac"]) > 0.02:
        text = ("SUSPECT: placement. A ran {}, B ran {}. Fix that first; "
                "nothing else gets blamed while the first rung "
                "differs.".format(a["place"], b["place"]))
    elif known("quant") and a["quant"] != b["quant"]:
        text = ("SUSPECT: quantization. Placement agrees, the weights do "
                "not ({} vs {}): different bytes per token, so the rates "
                "are not one series.".format(a["quant"], b["quant"]))
    elif known("ctx") and max(a["ctx"], b["ctx"]) >= 10 * min(a["ctx"],
                                                              b["ctx"]):
        text = ("SUSPECT: context size. Placement and quant agree; ctx "
                "{} against {} is an order of magnitude, and the KV "
                "cache scales with it.".format(a["ctx"], b["ctx"]))
    elif a["chip"] and b["chip"] and (a["chip"] != b["chip"]
                                      or a["ram"] != b["ram"]):
        text = ("SUSPECT: hardware. Every config variable both blocks "
                "carry agrees; the machines differ ({}, {} GB vs {}, {} "
                "GB). What is left is silicon, mostly memory bandwidth; "
                "a block cannot rank that.".format(
                    a["chip"], a["ram"], b["chip"], b["ram"]))
    else:
        if not (a["chip"] and b["chip"]):
            skipped.append("machine")
        text = ("NO SUSPECT: every variable both blocks carry agrees. "
                "What remains (background load, thermals, power mode, "
                "disk cache) does not print in a block; picchio will "
                "not guess.")
    minor = [k for k in ("engine", "threads", "os")
             if a[k] and b[k] and a[k] != b[k]]
    if minor and text.startswith(("SUSPECT: hardware", "NO SUSPECT")):
        text += (" Outside the ladder these differ too: "
                 + ", ".join(minor) + ".")
    return text, skipped


def render_compare(names, a, b):
    def cell(v):
        s = "unknown" if v is None else str(v)
        return s if len(s) <= 24 else s[:22] + ".."

    a, b = dict(a), dict(b)
    for x in (a, b):
        # a new-format block (it has a ctx line) with no bracket really
        # ran without extra args; an old block just cannot say
        x["args"] = x["args"] or ("none" if x["ctx"] else None)
        x["machine"] = "{}, {} GB".format(x["chip"], x["ram"]) \
            if x["chip"] else None
    rows = [(k, a[k], b[k]) for k in
            ("model", "quant", "engine", "place", "args", "ctx",
             "threads", "machine", "os")]
    if all(va == vb for _, va, vb in rows) and a["rates"] == b["rates"]:
        return ("picchio compare: A and B carry the same fingerprint "
                "and the same rates. Nothing to compare.")
    out = ["picchio compare", "A: " + names[0], "B: " + names[1], "",
           "{:<11}{:<26}{}".format("", "A", "B")]
    for label, va, vb in rows:
        same = va == vb and va is not None
        out.append("{:<11}{:<26}{}".format(
            label, cell(va), "same" if same else cell(vb)))
    note = a["row"] if a["row"] == b["row"] else \
        "A {}, B {}".format(a["row"], b["row"])
    out += ["", "rates ({}), tok/s:".format(note)]
    for i, lane in enumerate(("prefill", "decode", "wallclock")):
        va, vb = a["rates"][i], b["rates"][i]
        gap = "-"
        if va and vb:
            gap = "A {:.1f}x faster".format(va / vb) if va >= vb else \
                "B {:.1f}x faster".format(vb / va)
        out.append("  {:<11}{:>10}  {:>10}   {}".format(
            lane, "{:.1f}".format(va) if va else "n/a",
            "{:.1f}".format(vb) if vb else "n/a", gap))
    text, skipped = suspect_para(a, b)
    out += [""] + textwrap.wrap(text, width=WIDTH, subsequent_indent="  ")
    if skipped:
        out += textwrap.wrap("not judged, missing from one block: "
                             + ", ".join(skipped), width=WIDTH,
                             subsequent_indent="  ")
    return "\n".join(out)


def compare_cli(argv):
    if argv[:1] in (["-h"], ["--help"]) or len(argv) != 2:
        sys.exit("picchio compare: usage: picchio.py compare A.txt B.txt\n"
                 "each file holds one pasted verdict block (surrounding "
                 "forum text is fine)")
    blocks = []
    for path in argv:
        try:
            with open(path, errors="replace") as f:
                blk = parse_block(f.read())
        except OSError as e:
            sys.exit("picchio compare: {}".format(e))
        if blk is None:
            sys.exit("picchio compare: no verdict block in {} (need at "
                     "least the model line and a rates row)".format(path))
        blocks.append(blk)
    print(colorize(render_compare(argv, blocks[0], blocks[1])))


# ------------------------------------------------------------------ verify

def claim_shape(b):
    """Where a parsed block claims the work ran, from placement alone:
    'gpu', 'cpu', 'partial', or None when it reports no evidence."""
    if b["place_word"] in ("NO EVIDENCE", "EVIDENCE UNKNOWN"):
        return None
    frac = b["frac"]
    if b["place_word"] == "ENGAGED" or (frac is not None and frac >= 0.95):
        return "gpu"
    if b["place_word"] == "NOT ENGAGED" or (frac is not None and frac < 0.05):
        return "cpu"
    if b["place_word"] == "PARTIAL" or frac is not None:
        return "partial"
    return None


def verify_block(b):
    """Recomputes the physics a verdict block claims and checks the block
    agrees with itself. Every number in it is a shadow of one run:
    placement, the prefill/decode signature, the os meter and the
    headline each answer 'did the gpu do the work', and an honest block
    has all of them describing the same run. Returns (verdict, findings):
    PASS with no findings, or FLAG naming each physical contradiction.

    It cannot prove a block is real, since numbers can be faked so they
    agree; it proves only that a block contradicts itself, which is what
    fabrication and casual tampering almost always leave behind."""
    pf, dc, wc = b["rates"]
    claim = claim_shape(b)
    ratio = pf / dc if pf and dc else None
    f = []
    # 1. lane ordering is pure physics, hardware independent: prefill
    #    reads the whole prompt in one batched pass, decode writes one
    #    token at a time reading every weight each time, and wallclock
    #    spreads the generated tokens over load and prefill as well. On a
    #    single run prefill > decode > wallclock always holds; an
    #    inversion is a number that was typed, not measured.
    if pf and dc and dc >= pf:
        f.append("decode {:.1f} >= prefill {:.1f} tok/s: generation cannot "
                 "outrun prompt reading on one run".format(dc, pf))
    if dc and wc and wc >= dc:
        f.append("wallclock {:.1f} >= decode {:.1f} tok/s: wall time "
                 "includes load and prefill, it cannot be faster".format(
                     wc, dc))
    # 2. the prefill/decode ratio is a scale free signature of placement:
    #    a full-gpu run measures 20-44x on the calibrated machines, a cpu
    #    run 2-5x. A ratio that fights the placement claim is the
    #    ollama-ps-lies case (#7323 family), now caught in a static paste.
    if ratio is not None and claim == "gpu" and ratio < 5:
        f.append("claims full gpu but prefill is only {:.1f}x decode, a cpu "
                 "shaped ratio (a real gpu run is 20x+)".format(ratio))
    if ratio is not None and claim == "cpu" and ratio >= 15:
        f.append("claims no gpu but prefill is {:.1f}x decode, a gpu shaped "
                 "ratio a cpu run never reaches".format(ratio))
    # 3. the os meter is an independent witness, held against the claim
    #    only when it was sampled and the machine was idle enough to read;
    #    a block whose own os line already abstained is not judged on it
    if b["os_work"] is not None and b["os_note"] is None:
        if claim == "gpu" and b["os_work"] < 15:
            f.append("claims full gpu but its own os line saw the gpu at "
                     "{}% while the tokens were made".format(b["os_work"]))
        if claim == "cpu" and b["os_work"] >= 50:
            f.append("claims no gpu but its own os line saw the gpu busy at "
                     "{}% while the tokens were made".format(b["os_work"]))
    # 4. the headline must match the block's own placement line; a
    #    consistent body under a lying VERDICT word is the cheapest forgery
    if b["verdict"] == "HEALTHY" and claim in ("cpu", "partial"):
        f.append("headline says HEALTHY but the placement line says "
                 "{}".format(b["place_word"] or "not full gpu"))
    if b["verdict"] == "SILENT CPU FALLBACK" and claim == "gpu":
        f.append("headline says CPU FALLBACK but the placement line claims "
                 "the full gpu")
    return ("FLAG" if f else "PASS"), f


def render_verify(src, b, verdict, flags):
    pf, dc, wc = b["rates"]
    claim = claim_shape(b)
    shape = {"gpu": "full gpu", "cpu": "no gpu", "partial": "partial",
             None: "no placement evidence"}[claim]
    out = ["picchio verify: " + src,
           "  model     " + (b["model"] or "unknown"),
           "  claim     {} ({}), headline {}".format(
               b["place_word"] or "?", shape, b["verdict"] or "none")]
    if pf and dc:
        out.append("  signature prefill {:.1f} = {:.1f}x decode {:.1f}, "
                   "wallclock {}".format(
                       pf, pf / dc, dc,
                       "{:.1f}".format(wc) if wc else "n/a"))
    if b["os_raw"]:
        out.append("  os        " + b["os_raw"])
    if verdict == "PASS":
        witnessed = b["os_work"] is not None and b["os_note"] is None
        out.append("VERDICT: PASS. placement, the timing signature"
                   + (" and the os meter" if witnessed else "")
                   + " all describe the same run.")
    else:
        out.append("VERDICT: FLAG. {} physical contradiction{} in this "
                   "block:".format(len(flags),
                                   "" if len(flags) == 1 else "s"))
        for fl in flags:
            out.extend(textwrap.wrap(fl, width=WIDTH, initial_indent="  - ",
                                     subsequent_indent="    "))
        out.append("This block contradicts itself; do not trust its numbers "
                   "as one run.")
    return "\n".join(out)


def verify_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio.py verify [FILE]\n"
              "re-derive the physics a pasted verdict block claims, and\n"
              "flag it when placement, the prefill/decode signature, the\n"
              "os meter and the headline do not describe the same run.\n"
              "reads the block from FILE, or from stdin when none is given.")
        sys.exit(0)
    src = argv[0] if argv and argv[0] != "-" else None
    if src:
        try:
            text = open(src, errors="replace").read()
        except OSError as e:
            sys.exit("picchio verify: {}".format(e))
    else:
        text = sys.stdin.read()
        src = "pasted block"
    b = parse_block(text)
    if b is None:
        sys.stderr.write("picchio verify: no verdict block found in {} (need "
                         "the model line and a rates row).\n".format(src))
        sys.exit(2)
    verdict, flags = verify_block(b)
    print(colorize(render_verify(src, b, verdict, flags)))
    # reuse the measure exit map: a self-consistent block is 0, a block
    # whose sources fight is CONFLICTING EVIDENCE (5), the same code a
    # live run gets when two sources disagree
    sys.exit(0 if verdict == "PASS" else 5)


# ------------------------------------------------------------------- watch
#
# watch reads placement the engine-free way: it does not parse anyone's
# stderr, it points the OS meter at a running process or the whole GPU
# and reports what the silicon is doing. That makes it engine agnostic:
# MLX, LM Studio, vLLM, a raw torch script, anything that generates can
# be watched. ioreg meters the whole GPU, not one process, so watch
# never claims per-process precision: it reports machine level truth and
# says so, exactly the abstain discipline the measure-mode vote already
# uses on a busy desktop.

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by another user


def proc_name(pid):
    out = _cmd_out(["ps", "-p", str(pid), "-o", "comm="]).splitlines()
    return os.path.basename(out[0]) if out and out[0] else "?"


def ollama_loaded():
    """The first model ollama currently has resident, or None with a
    reason string. watch uses it only as a label for what is running."""
    if not ollama_reachable():
        return None, "no ollama is answering at {}".format(OLLAMA_HOST)
    try:
        models = ollama_api("/api/ps", timeout=5).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return None, "ollama did not answer /api/ps"
    if not models:
        return None, "ollama is running but no model is loaded"
    return models[0].get("name") or models[0].get("model") or "?", None


def watch_summary(samples):
    """Distills a raw sample window into the machine level numbers watch
    reports: no baseline step (the process is already running, there is
    no clean before), just what the GPU did over the window."""
    dev = [s["dev"] for s in samples if s.get("dev") is not None]
    mem = [s["mem"] for s in samples if s.get("mem") is not None]
    watts = [s.get("gpu_w") for s in samples]
    return {
        "n": len(samples),
        "secs": samples[-1]["t"] - samples[0]["t"] if len(samples) >= 2
        else 0.0,
        "work_med": _med(dev), "work_peak": max(dev) if dev else None,
        "work_min": min(dev) if dev else None,
        "mem_gib": max(mem) / 1024 ** 3 if mem else None,
        "watts": _med(watts), "throttled": False,
    }


def watch_verdict(summ, ctx):
    """Machine level placement read: is the GPU doing the work. ctx is a
    label for what is being watched (a process, an ollama model) or None
    for the whole machine; when set, the whole-GPU caveat is spelled out
    rather than pretending the number belongs to that one job."""
    wm = summ["work_med"]
    if wm is None:
        return "GPU UNREADABLE", "the gpu meter returned no usable samples."
    w = ", {:.1f} W".format(summ["watts"]) if summ["watts"] is not None else ""
    if wm >= 50:
        para = ("something is running kernels on the gpu (work {:.0f}% "
                "median, peak {:.0f}%{}).".format(wm, summ["work_peak"], w))
        if ctx:
            para += (" ioreg meters the whole gpu, so this is machine level, "
                     "not pinned to {}.".format(ctx))
            if summ["work_min"] is not None and summ["work_min"] < 15:
                para += " It fell idle between bursts, consistent with one job."
        return "GPU BUSY", para
    if wm < 15:
        para = "the gpu ran at {:.0f}% median over the window.".format(wm)
        para += (" If {} is generating tokens now, it is doing it on the cpu, "
                 "not the gpu.".format(ctx) if ctx
                 else " Nothing is driving the gpu right now.")
        return "GPU IDLE", para
    para = ("the gpu is lightly used (work {:.0f}% median, peak {:.0f}%{}): "
            "partial offload, or another job sharing it.".format(
                wm, summ["work_peak"], w))
    if ctx:
        para += " ioreg is whole-gpu; machine level only."
    return "GPU MIXED", para


def render_watch(ctx, summ, state, para):
    out = ["picchio watch" + (": " + ctx if ctx else "")]
    out.append("  window   {:.1f} s, {} samples at {:.0f} Hz  (whole "
               "gpu)".format(summ["secs"], summ["n"], TELE_HZ))
    parts = []
    if summ["work_med"] is not None:
        parts.append("work {:.0f}% median".format(summ["work_med"]))
    if summ["work_peak"] is not None:
        parts.append("peak {:.0f}%".format(summ["work_peak"]))
    if summ["watts"] is not None:
        parts.append("{:.1f} W".format(summ["watts"]))
    if summ["throttled"]:
        parts.append("throttled")
    if parts:
        out.append("  gpu      " + ", ".join(parts))
    if summ["mem_gib"] is not None:
        out.append("  memory   {:.1f} GiB in use by the gpu".format(
            summ["mem_gib"]))
    out += textwrap.wrap("{}: {}".format(state, para), width=WIDTH,
                         subsequent_indent="  ")
    return "\n".join(out)


def watch(pid=None, engine=None, duration=None):
    ctx = None
    if engine is not None:
        if engine != "ollama":
            sys.exit("picchio watch: only --engine ollama is supported "
                     "(any other engine: give its pid, or just watch the "
                     "whole gpu with no argument).")
        name, why = ollama_loaded()
        if name is None:
            sys.exit("picchio watch: {}. Load a model and generate, then "
                     "watch.".format(why))
        ctx = "ollama model " + name
    if pid is not None:
        if not pid_alive(pid):
            sys.exit("picchio watch: no process with pid {}.".format(pid))
        ctx = "{} (pid {})".format(proc_name(pid), pid)
    sampler = telemetry_start()
    if not isinstance(sampler, GpuSampler):
        sys.exit("picchio watch: no gpu meter here ({}). watch needs the "
                 "macos ioreg meter; on other platforms there is no engine "
                 "free placement signal yet.".format(sampler.get("off", "?")))
    if sampler._backend is not None:
        # the nvml meter feeds measure mode only for now; watch on
        # linux is a separate milestone with its own calibration
        sampler.stop()
        sys.exit("picchio watch: watch is ioreg only for now; on linux "
                 "the nvml meter runs inside measure mode.")
    # window: an explicit --for wins; otherwise watch until the pid exits
    # (capped), or a short fixed window for the whole-gpu snapshot
    if duration is None:
        duration = 3600.0 if pid is not None else 6.0
    sys.stderr.write("picchio watch: sampling the gpu{}{} ...\n".format(
        " while " + ctx if ctx else "",
        "" if pid is not None and duration >= 3600 else
        " for {:.0f} s".format(duration)))
    deadline = time.monotonic() + duration
    try:
        while True:
            now = time.monotonic()
            if now >= deadline or (pid is not None and not pid_alive(pid)):
                break
            time.sleep(min(0.25, deadline - now))
    except KeyboardInterrupt:
        pass
    sampler.stop()
    summ = watch_summary(sampler.samples)
    summ["throttled"] = sampler._hot or thermal_raised()
    state, para = watch_verdict(summ, ctx)
    print(colorize(render_watch(ctx, summ, state, para)))
    # reuse the measure exit map's meaning: the gpu doing the work is 0,
    # the gpu sitting idle while tokens are made is the fallback code (4)
    sys.exit({"GPU IDLE": 4}.get(state, 0))


def watch_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio.py watch [PID] [--engine ollama] [--for SEC]\n"
              "point the os gpu meter at a running inference process (or\n"
              "the whole gpu) and report whether the gpu is doing the work,\n"
              "without parsing any engine's output. engine agnostic: works\n"
              "for mlx, lm studio, anything. macOS only (needs ioreg).")
        sys.exit(0)
    pid = engine = dur = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--for" and i + 1 < len(argv):
            try:
                dur = float(argv[i + 1])
            except ValueError:
                sys.exit("picchio watch: --for wants a number of seconds.")
            i += 2
        elif a == "--engine" and i + 1 < len(argv):
            engine, i = argv[i + 1], i + 2
        elif a.isdigit():
            pid, i = int(a), i + 1
        else:
            sys.exit("picchio watch: unexpected argument {!r}.\nusage: "
                     "picchio.py watch [PID] [--engine ollama] "
                     "[--for SEC]".format(a))
    watch(pid, engine, dur)


# ------------------------------------------------------------------ monitor
#
# measure and server mode each take one snapshot; a setup that runs on the
# GPU now can fall to the CPU an hour later, on a reload the http api never
# announces, and the next snapshot you happen to take is the only place you
# would ever see it. monitor closes that window: it sends one controlled
# probe on a fixed interval to a running llama-server, reads the per
# request prefill and decode timings the server already returns, and
# classifies each probe by the same ratio signature the block votes on
# (prefill under 5x decode is cpu shaped, 15x and over gpu shaped, the band
# between is unsure). Each probe is one line; a probe that flips the running
# placement prints a louder line, because a GPU that comes and goes between
# requests is exactly the failure a single snapshot cannot catch. It
# launches nothing and kills nothing: the server is the user's, monitor
# only knocks on it on a timer. The probe reuses measure's fixed prompt, so
# a short prompt can never make prefill look slow (the trap the whole tool
# exists to warn about); that is why it reads its own probes and not the
# user's variable length traffic.

MON_CPU_RATIO = 5.0     # prefill under this many x decode reads cpu shaped
MON_GPU_RATIO = 15.0    # this and over reads gpu shaped; between is unsure
MON_EVERY_S = 30.0      # default seconds between probes; each probe is one
                        # full BENCH_PROMPT completion, so a shorter gap
                        # puts more real load on the server being watched
MON_TAG = {"OK": "ENGAGED", "FLAG": "NOT ENGAGED",
           "WATCH": "UNSURE", "NODATA": "NO DATA"}


def monitor_classify(prefill, decode):
    """One probe's verdict from its two rates, the same signature the
    server block votes on. Returns (state, ratio): OK when the shape is
    gpu, FLAG when it is cpu, WATCH in the unsure band, NODATA when a rate
    is missing (a probe that came back without usable timings convicts
    nobody)."""
    if not prefill or not decode:
        return "NODATA", None
    ratio = prefill / decode
    if ratio < MON_CPU_RATIO:
        return "FLAG", ratio
    if ratio >= MON_GPU_RATIO:
        return "OK", ratio
    return "WATCH", ratio


def monitor_summarize(events):
    """The exit summary from the probe log. events is a list of (state,
    ratio) in order. Counts the two decisive states, the transitions
    between them (OK<->FLAG, the intermittent signal a snapshot misses),
    and the worst prefill/decode ratio seen."""
    ok = sum(1 for s, _ in events if s == "OK")
    flag = sum(1 for s, _ in events if s == "FLAG")
    decisive = [s for s, _ in events if s in ("OK", "FLAG")]
    trans = sum(1 for a, b in zip(decisive, decisive[1:]) if a != b)
    ratios = [r for _, r in events if r is not None]
    return {"n": len(events), "ok": ok, "flag": flag, "transitions": trans,
            "worst_ratio": min(ratios) if ratios else None}


def monitor_line(stamp, i, state, ratio, prefill, decode):
    """One compact status line per probe, pasteable, colorized by the
    monitor branch in colorize()."""
    head = "picchio monitor: {} probe {:<3} {}".format(
        stamp, i, MON_TAG[state])
    if state == "NODATA":
        return head + "  the server returned no usable timings"
    return "{}  prefill {}, decode {} ({:.1f}x)".format(
        head, fmt_rate(prefill), fmt_rate(decode), ratio)


def monitor_summary_line(summ):
    """The one line printed when monitor stops: what it saw over the whole
    session, and the verdict that sets the exit code."""
    if summ["n"] == 0:
        return "picchio monitor: no probes completed."
    parts = ["{} probes".format(summ["n"]),
             "{} engaged".format(summ["ok"]),
             "{} on cpu".format(summ["flag"])]
    if summ["transitions"]:
        parts.append("{} placement change(s)".format(summ["transitions"]))
    if summ["worst_ratio"] is not None:
        parts.append("worst prefill {:.1f}x decode".format(
            summ["worst_ratio"]))
    verdict = "SILENT CPU FALLBACK seen" if summ["flag"] \
        else "ENGAGED throughout"
    return "picchio monitor: {} - {}".format(verdict, ", ".join(parts))


def _mon_secs(flag, val):
    try:
        s = float(val)
    except ValueError:
        sys.exit("picchio monitor: {} wants a number of seconds.".format(
            flag))
    if s <= 0:
        sys.exit("picchio monitor: {} wants a positive number.".format(flag))
    return s


def _monitor_wait(t0, every, deadline):
    """Sleep out the rest of one interval after a probe, but never past
    the --for deadline. Returns False once the deadline has arrived so the
    loop stops on time instead of one probe late."""
    target = t0 + every
    if deadline is not None:
        target = min(target, deadline)
    time.sleep(max(0.0, target - time.monotonic()))
    return deadline is None or time.monotonic() < deadline


def monitor(url, every=MON_EVERY_S, duration=None, keep_dir=None):
    """Probe a running llama-server on a timer and flag any probe whose
    prefill/decode signature goes cpu shaped. Speaks one line per probe and
    a louder line whenever the placement flips, never launches or signals
    the server, and exits 4 the moment any probe caught the gpu not doing
    the work (0 if it held the whole session)."""
    ok, why = server_health(url)
    if not ok:
        sys.exit("picchio monitor: " + why)
    ctx = server_ctx(url)
    sys.stderr.write(
        "picchio monitor: probing {} every {:.0f} s (ctx {}); "
        "ctrl-c to stop\n".format(url, every, ctx))
    events, last_decisive, i = [], None, 0
    deadline = (time.monotonic() + duration) if duration else None
    try:
        while deadline is None or time.monotonic() < deadline:
            i += 1
            t0 = time.monotonic()
            lp = os.path.join(keep_dir, "probe{}.response.json".format(i)) \
                if keep_dir else None
            try:
                p = run_server_pass(url, lp)
            except (urllib.error.URLError, OSError, ValueError) as e:
                # a server that stopped answering is an event worth a line,
                # but not a cpu conviction; keep the timer running so a
                # restart is picked up on the next tick
                sys.stderr.write("picchio monitor: probe {} could not reach "
                                 "the server: {}\n".format(i, e))
                if not _monitor_wait(t0, every, deadline):
                    break
                continue
            prefill, decode = p["prefill_toks"], p["decode_toks"]
            state, ratio = monitor_classify(prefill, decode)
            events.append((state, ratio))
            sys.stderr.write(colorize(monitor_line(
                time.strftime("%H:%M:%S"), i, state, ratio, prefill, decode),
                sys.stderr) + "\n")
            if state in ("OK", "FLAG"):
                if last_decisive and state != last_decisive:
                    sys.stderr.write(colorize(
                        "picchio monitor: placement changed {} -> {} at "
                        "probe {}".format(MON_TAG[last_decisive],
                                          MON_TAG[state], i),
                        sys.stderr) + "\n")
                last_decisive = state
            if not _monitor_wait(t0, every, deadline):
                break
    except KeyboardInterrupt:
        sys.stderr.write("\n")
    summ = monitor_summarize(events)
    sys.stderr.write(colorize(monitor_summary_line(summ), sys.stderr) + "\n")
    sys.exit(4 if summ["flag"] else 0)


def monitor_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio.py monitor URL [--every SEC] [--for SEC] "
              "[--keep-logs DIR]\n"
              "probe a running llama-server on an interval and flag any\n"
              "probe whose prefill/decode signature goes cpu shaped: the\n"
              "intermittent fallback a single snapshot cannot see. never\n"
              "launches or kills the server.")
        sys.exit(0)
    url = keep = None
    every, dur, i = MON_EVERY_S, None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--every" and i + 1 < len(argv):
            every, i = _mon_secs("--every", argv[i + 1]), i + 2
        elif a == "--for" and i + 1 < len(argv):
            dur, i = _mon_secs("--for", argv[i + 1]), i + 2
        elif a == "--keep-logs" and i + 1 < len(argv):
            keep = argv[i + 1]
            os.makedirs(keep, exist_ok=True)
            i += 2
        elif a.startswith(("http://", "https://")) and url is None:
            url, i = a, i + 1
        else:
            sys.exit("picchio monitor: unexpected argument {!r}.\nusage: "
                     "picchio.py monitor URL [--every SEC] [--for SEC] "
                     "[--keep-logs DIR]".format(a))
    if url is None:
        sys.exit("picchio monitor: give the url of a running llama-server, "
                 "e.g. picchio.py monitor http://127.0.0.1:8080")
    monitor(url, every, dur, keep)


# --------------------------------------------------------------- ctx sweep
#
# One tok/s number is measured at one context depth, almost always a
# short one, and quoted as if it held at any length. It does not: every
# token decode generates attends to the whole kv cache, so decode slows
# as the context fills. The sweep re-measures the three lanes at a few
# ctx depths, each with a prompt long enough to actually fill that depth
# (a short prompt at -c 32768 fills nothing and would just measure the 4k
# number again), and reports the decay slope. It answers a question the
# forums do not: what does your decode rate do at 32k that it did not at
# 4k.

def resolve_engine(model, bin_):
    """llama.cpp-vs-ollama-vs-server resolution shared by measure and
    sweep: an existing file is llama.cpp, an http(s) url is a running
    llama-server, a bare tag is ollama, a missing path is an error
    (never quietly retried as a tag). Returns (mode, binpath,
    engine_str, model_name); for a server the binpath slot carries the
    url, since there is no binary to find."""
    if model.startswith(("http://", "https://")):
        url = model.rstrip("/")
        ok, why = server_health(url)
        if not ok:
            sys.exit("picchio: " + why)
        props = server_props(url)
        name = os.path.basename(props.get("model_path") or "") \
            or props.get("model_alias") or url
        build = props.get("build_info") or "?"
        return "server", url, "llama-server " + str(build), name
    if os.path.isfile(model):
        binpath = find_binary(bin_)
        name = os.path.basename(model)
        try:
            # the header's own name beats an uninformative file name
            # (a 4090 fixture block read "model.gguf"); when the header
            # name lacks the quant token the file name carries, the
            # token rides along so the compare fingerprint stays whole
            gname = gguf_meta(model).get("general.name")
            if gname:
                qm = RE_QUANT.search(name)
                if qm and not RE_QUANT.search(gname):
                    gname += " " + qm.group(1).upper()
                name = str(gname)
        except (ValueError, struct.error, KeyError, OSError):
            pass
        return ("llama.cpp", binpath,
                "llama.cpp " + engine_version(binpath), name)
    if not looks_like_tag(model):
        sys.exit("picchio: no such file: {}\nRun picchio with no arguments "
                 "to see the models on this machine.".format(model))
    ver = ollama_reachable()
    if not ver:
        sys.exit("picchio: {!r} looks like an ollama tag, but no ollama "
                 "answered at {}.\nStart ollama, or give a .gguf path.".format(
                     model, OLLAMA_HOST))
    if not ollama_has_model(model):
        sys.exit("picchio: ollama at {} does not know the model {!r}.".format(
            OLLAMA_HOST, model))
    return "ollama", None, "ollama " + ver, model


def sweep_prompt(target_tokens):
    """A prompt long enough to fill about target_tokens of context, so
    decode is measured at real kv depth. English runs a little over one
    token per word here, so the paragraph is repeated until the word
    count crosses the target; the block reports the depth the engine
    actually reached, not this estimate."""
    words = len(_PARA.split())
    reps = max(1, int(target_tokens / (words * 1.25)))
    return "".join("Passage {}. {}".format(i + 1, _PARA) for i in range(reps))


def parse_tiers(spec):
    tiers = sorted({int(t) for t in spec.split(",")
                    if t.strip().isdigit() and int(t) > 0})
    if len(tiers) < 2:
        sys.exit("picchio: --ctx-sweep needs at least two ctx sizes, "
                 "e.g. --ctx-sweep 4096,32768")
    return tiers


def ctx_sweep(model, mode, binpath, engine_str, model_name, tiers, passes, lp):
    """Re-measures the three lanes at each ctx tier, each tier fed a
    prompt sized to fill it, so decode is read at real kv depth. Returns
    one row per tier: the depth actually reached and the warm median
    lanes. The first pass of each tier is the cold one and is dropped
    from the warm median, exactly as measure mode does it."""
    rows = []
    for ctx in tiers:
        prompt = sweep_prompt(int(ctx * 0.7))  # leave headroom for 128 gen
        ps = []
        for i in range(passes):
            sys.stderr.write("picchio: ctx {} pass {}/{}{} ...\n".format(
                ctx, i + 1, passes, " (includes cold load)" if i == 0 else ""))
            if mode == "llama.cpp":
                p = run_llama_pass(
                    binpath, model, [],
                    lp("ctx{}.pass{}.stderr.txt".format(ctx, i + 1)),
                    prompt=prompt, ctx=ctx)
            else:
                p, _ = run_ollama_pass(
                    model, lp("ctx{}.pass{}.response.json".format(ctx, i + 1)),
                    prompt=prompt, ctx=ctx)
            # wall_s is measured by picchio, not in the engine log; persist
            # it per pass so the sweep table can be replayed like a verdict
            keep_log(lp("ctx{}.pass{}.meta.json".format(ctx, i + 1)),
                     json.dumps({"wall_s": p["wall_s"]}, indent=1))
            ps.append(p)
        rep = build_rep(ps)
        rows.append({"ctx": ctx, "depth": rep.get("prompt_tokens"),
                     "prefill": rep["prefill_toks"],
                     "decode": rep["decode_toks"],
                     "wallclock": rep["wallclock_toks"]})
    keep_log(lp("sweep.meta.json"), json.dumps(
        {"engine": engine_str, "model_name": model_name, "mode": mode,
         "tiers": tiers, "passes": passes}, indent=1))
    return rows


def sweep_slope(rows):
    """The decay sentence: decode from the shallowest to the deepest tier
    reached, the lane long context actually taxes. None when either end
    is unmeasured."""
    lo, hi = rows[0], rows[-1]
    if not (lo["decode"] and hi["decode"] and lo["depth"] and hi["depth"]):
        return None
    drop = (1 - hi["decode"] / lo["decode"]) * 100
    span = hi["depth"] / lo["depth"]
    ends = "{} to {} tokens ({:.0f}x deeper): {:.1f} -> {:.1f} tok/s".format(
        lo["depth"], hi["depth"], span, lo["decode"], hi["decode"])
    if drop >= 5:
        return ("decode fell {:.0f}% from {}. Long context is not free; the "
                "kv cache taxes every token you generate.".format(drop, ends))
    if drop >= -5:
        return ("decode held within {:.0f}% from {}. Here weight bandwidth "
                "dominates and the kv cache barely shows.".format(
                    abs(drop), ends))
    return ("decode read {:.0f}% faster from {}: that is measurement noise, "
            "not a real speedup at depth.".format(-drop, ends))


def render_sweep(mach, engine_str, model_name, rows):
    out = ["ctx sweep  " + ", ".join(x for x in (model_name, engine_str) if x),
           "{:<15}{:>12}  {:>12}  {:>12}".format(
               "depth   ctx", "prefill", "decode", "wallclock")]
    for r in rows:
        out.append("{:<15}{:>12}  {:>12}  {:>12}".format(
            "{:>6}  {}".format(r["depth"] if r["depth"] else "?", r["ctx"]),
            fmt_rate(r["prefill"]), fmt_rate(r["decode"]),
            fmt_rate(r["wallclock"])))
    slope = sweep_slope(rows)
    if slope:
        out += textwrap.wrap("SLOPE: " + slope, width=WIDTH,
                             subsequent_indent="  ")
    out.append("-- picchio v{} ctx-sweep on {}, {} GB, {}".format(
        VERSION, mach["chip"], mach["ram_gb"] or "?", mach["os"]))
    return "\n".join(out)


# ------------------------------------------------------- plan (capacity)
#
# picchio plan answers the question people ask before the download
# finishes: will this model fit this machine, and roughly how fast will
# it decode. The fit half is static and always available: the GGUF
# header carries the geometry, and the account it feeds matched the
# engine's own allocations exactly on both local models. The speed half
# is only ever a projection of this machine's own last measured run;
# with no run cached there is no number at all, because an estimate
# with no measurement behind it is a guess wearing digits. Nothing plan
# prints is a verdict block: no 15 line protocol, no mp1 footer, so a
# projection can never be pasted somewhere a measurement belongs.

GGUF_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
              6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_meta_stream(f):
    """The GGUF v2/v3 header key value table, scalars and strings only
    (arrays are read past and dropped): magic, version, tensor count,
    kv count, then typed pairs. Layout and every key name used
    downstream were checked against the two local model files and
    against ollama's /api/show mirror of the same table."""
    if f.read(4) != b"GGUF":
        raise ValueError("not a gguf file (magic mismatch)")
    version = struct.unpack("<I", f.read(4))[0]
    if version < 2:
        raise ValueError("gguf v{} predates the v2 layout".format(version))
    # tensor count: no part of the kv account, but the id walk resumes
    # right after this function, so it rides along under a private key
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    if n_kv > 65536:
        raise ValueError("gguf header claims {} keys".format(n_kv))

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 1 << 24:
            raise ValueError("gguf string of {} bytes".format(n))
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            it = struct.unpack("<I", f.read(4))[0]
            cnt = struct.unpack("<Q", f.read(8))[0]
            if it == 8:
                for _ in range(cnt):
                    rstr()
            elif it == 9:
                raise ValueError("nested gguf array")
            else:
                f.seek(struct.calcsize(GGUF_TYPES[it]) * cnt, 1)
            return None  # array values feed nothing in the account
        fmt = GGUF_TYPES[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    out = {"__tensor_count": n_tensors}
    for _ in range(n_kv):
        key = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        v = rval(t)
        if v is not None:
            out[key] = v
    return out


def gguf_meta(path):
    with open(path, "rb") as f:
        return gguf_meta_stream(f)


def _arch_get(meta, key):
    arch = meta.get("general.architecture")
    return meta.get("{}.{}".format(arch, key)) if arch else None


def plan_is_moe(meta):
    return bool(_arch_get(meta, "expert_count"))


def kv_account(meta, ctx=CTX):
    """(kv bytes at ctx, note). Formula: ctx x attention layers x kv
    heads x (key length + value length) x 2 bytes of f16. Hybrid
    attention models mark every Nth layer as full attention
    (full_attention_interval) and the rest hold constant state, so the
    interval divides the layer count; honoring it lands this exactly
    on the engine's own llama_kv_cache allocation for both local
    models (128.00 MiB on the 9B, 80.00 MiB on the 35B MoE, ctx 4096,
    the 9B line committed in examples/raw/healthy-metal). Experts
    change the ffn, not the kv, so MoE needs no special case here.
    When key/value length are absent the classic head_dim fallback is
    embedding_length over head_count."""
    if not meta.get("general.architecture"):
        return None, "header lacks general.architecture"
    blocks = _arch_get(meta, "block_count")
    heads = _arch_get(meta, "attention.head_count")
    heads_kv = _arch_get(meta, "attention.head_count_kv") or heads
    klen = _arch_get(meta, "attention.key_length")
    vlen = _arch_get(meta, "attention.value_length")
    if (not klen or not vlen) and _arch_get(meta, "embedding_length") \
            and heads:
        klen = vlen = int(_arch_get(meta, "embedding_length")) // int(heads)
    if not (blocks and heads_kv and klen and vlen):
        return None, "header lacks the kv geometry keys"
    interval = int(_arch_get(meta, "full_attention_interval") or 1)
    att = max(1, int(blocks) // max(1, interval))
    note = "at ctx {}".format(ctx)
    if interval > 1:
        note += ", {} of {} layers attend".format(att, blocks)
    return int(ctx) * att * int(heads_kv) * (int(klen) + int(vlen)) * 2, note


PLAN_COMPUTE = 512 * 1024 ** 2  # the graph buffer: sched_reserve
# measured 505.02 MiB on the 9B and 493.00 MiB on the 35B here, so a
# flat half GiB stands in for what the header cannot predict


def plan_budget(mach):
    """(budget bytes, label). On macOS the wall is the metal working
    set, about 0.78 of ram: the engine itself reported 25558 MiB free
    on the idle 32 GB test machine (the MiB-free figure in
    examples/raw/healthy-metal). Elsewhere no fraction has been
    calibrated yet, so whole ram is the bar and the label says the
    check is ram only."""
    ram = (mach["ram_gb"] or 0) * 1024 ** 3
    if not ram:
        return None, "ram size unknown"
    if platform.system() == "Darwin":
        return int(ram * 0.78), "metal working set, 0.78 of {} GB ram" \
            .format(mach["ram_gb"])
    return ram, "system ram only, gpu memory not judged"


def plan_state(need, budget):
    """fits / tight / no. The 35B MoE measured HEALTHY fully offloaded
    at 85% of this budget (22.1 GB of weights on the 32 GB machine),
    so fits runs to 0.95; past 1.05 even an idle machine has no room
    left to find."""
    r = need / budget
    if r <= 0.95:
        return "fits"
    if r <= 1.05:
        return "tight"
    return "no"


def plan_speed_source(cache):
    """(bytes/s bandwidth, provenance) or (None, refusal). The one
    legal source for a speed figure here is this machine's own last
    measured run: warm decode times file size, the same arithmetic the
    README derives effective bandwidth with. No cached run means no
    number at all, and a mixture of experts cannot calibrate it: its
    decode reads only the active experts, so decode times file size
    overstates the bandwidth several fold."""
    if not cache or not cache.get("model_bytes") \
            or not (cache.get("rates") or {}).get("decode"):
        return None, ("speed: not calibrated, no measured run cached "
                      "on this machine. Run a diagnosis once (python3 "
                      "picchio.py MODEL) and plan gains an estimated "
                      "decode column from that run's bandwidth.")
    if cache.get("moe"):
        return None, ("speed: the cached run ({}) is a mixture of "
                      "experts, and its bandwidth arithmetic does not "
                      "transfer. Diagnose a dense model once for the "
                      "estimate.".format(cache.get("model_name", "?")))
    bw = cache["rates"]["decode"] * cache["model_bytes"]
    return bw, "calibrated by {} at {:.1f} tok/s decode".format(
        cache.get("model_name", "?"), cache["rates"]["decode"])


def plan_est_decode(bw, file_bytes, moe):
    """Estimated decode for one target, or None: a MoE target is never
    priced (the file is not what each token reads)."""
    if bw is None or not file_bytes or moe:
        return None
    return bw / file_bytes


def _gib(n):
    return "{:.1f} GiB".format(n / 1024 ** 3)


def plan_target(arg):
    """Resolve one plan argument into (name, file_bytes, meta, note):
    a .gguf path is read directly, an ollama tag through /api/show
    (model_info mirrors the same header keys) plus /api/tags for the
    blob size. meta is None when unreadable, and note says why."""
    if os.path.isfile(arg):
        try:
            return (os.path.basename(arg), os.path.getsize(arg),
                    gguf_meta(arg), None)
        except (ValueError, struct.error, KeyError, OSError) as e:
            return (os.path.basename(arg), os.path.getsize(arg),
                    None, str(e))
    if not looks_like_tag(arg):
        sys.exit("picchio plan: no such file: {}".format(arg))
    if not ollama_reachable():
        sys.exit("picchio plan: {!r} looks like an ollama tag, but no "
                 "ollama answered at {}.".format(arg, OLLAMA_HOST))
    try:
        show = ollama_api("/api/show", {"model": arg}, timeout=15)
    except (urllib.error.URLError, OSError, ValueError):
        sys.exit("picchio plan: ollama at {} does not know the model "
                 "{!r}.".format(OLLAMA_HOST, arg))
    size = None
    try:
        for m in ollama_api("/api/tags", timeout=5).get("models", []):
            if m.get("name") == arg or m.get("model") == arg:
                size = m.get("size")
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return arg, size, show.get("model_info") or {}, None


def plan_row(name, file_bytes, meta, note, budget, bw):
    """One accounted row: need, state, estimate; honest holes where
    the evidence is missing."""
    if file_bytes is None:
        return {"name": name, "need": None, "state": "not judged",
                "est": None, "moe": False,
                "note": note or "no size available"}
    kv, kv_note = kv_account(meta) if meta else (None, note or "?")
    need = file_bytes + (kv or 0) + PLAN_COMPUTE
    moe = plan_is_moe(meta) if meta else False
    state = plan_state(need, budget) if budget else "not judged"
    return {"name": name, "need": need, "state": state, "moe": moe,
            "est": plan_est_decode(bw, file_bytes, moe),
            "kv": kv, "kv_note": kv_note, "file": file_bytes,
            "note": None if meta else (note or "header unreadable")}


def render_plan_one(row, budget, blabel, bw, speed_note):
    out = ["picchio plan: " + row["name"]]
    if row["need"] is None:
        out.append("  " + row["note"])
        return "\n".join(out)
    out.append("  weights   {:>10}   the file itself".format(
        _gib(row["file"])))
    if row["kv"] is not None:
        out.append("  kv cache  {:>10}   {}".format(_gib(row["kv"]),
                                                    row["kv_note"]))
    else:
        out.append("  kv cache  {:>10}   not counted: {}".format(
            "?", row.get("kv_note") or row.get("note") or "?"))
    out.append("  compute   {:>10}   graph buffer, measured constant"
               .format(_gib(PLAN_COMPUTE)))
    out.append("  need      {:>10}".format(_gib(row["need"])))
    if budget:
        out.append("  budget    {:>10}   {}".format(_gib(budget), blabel))
        out.append("  verdict   {:>10}   {:.0f}% of budget".format(
            row["state"], 100.0 * row["need"] / budget))
    else:
        out.append("  verdict   not judged   " + blabel)
    if row["note"]:
        out.append("  note: " + row["note"])
    if row["moe"]:
        out.append("  speed: no estimate for a mixture of experts; each")
        out.append("  token reads only the active experts, so file size")
        out.append("  arithmetic would lie about it.")
    elif row["est"] is not None:
        out.append("  est decode  ~{:.1f} tok/s   estimate, not a "
                   "measurement".format(row["est"]))
        out += textwrap.wrap(speed_note, width=WIDTH - 4,
                             initial_indent="    ",
                             subsequent_indent="    ")
    if row["est"] is None and not row["moe"]:
        out += textwrap.wrap(speed_note, width=WIDTH,
                             initial_indent="  ", subsequent_indent="  ")
    return "\n".join(out)


def render_plan_scan(rows, budget, blabel, bw, speed_note):
    out = ["picchio plan: {} model{} on this machine".format(
        len(rows), "" if len(rows) == 1 else "s")]
    if budget:
        out.append("budget {} ({})".format(_gib(budget), blabel))
        out.append("kv counted at ctx {}".format(CTX))
    else:
        out.append("budget not judged: " + blabel)
    out.append("")
    calibrated = bw is not None
    head = "  {:<30}{:>9}   {:<5}".format("model", "need", "fit")
    if calibrated:
        head += "  est decode"
    out.append(head.rstrip())
    for r in rows:
        name = r["name"] if len(r["name"]) <= 30 else r["name"][:28] + ".."
        line = "  {:<30}{:>9}   {:<5}".format(
            name, _gib(r["need"]) if r["need"] else "?", r["state"])
        if calibrated:
            if r["est"] is not None:
                line += "  ~{:.1f} tok/s".format(r["est"])
            elif r["moe"]:
                line += "  n/a (moe)"
            else:
                line += "  n/a"
        out.append(line.rstrip())
    out.append("")
    if calibrated:
        out += textwrap.wrap("every est decode figure is an estimate "
                             "projected from one measured run ({}), not "
                             "a measurement".format(speed_note),
                             width=WIDTH, subsequent_indent="  ")
    else:
        out += textwrap.wrap(speed_note, width=WIDTH,
                             subsequent_indent="  ")
    return "\n".join(out)


def plan_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio.py plan [MODEL]\n"
              "the capacity account before you download or load: will it\n"
              "fit (gguf header geometry against this machine's memory\n"
              "budget), and, once one real diagnosis has been run here,\n"
              "an estimated decode rate. With no MODEL, accounts every\n"
              "model found on this machine. Estimates are labeled and\n"
              "never appear in a verdict block.")
        sys.exit(0)
    if len(argv) > 1:
        sys.exit("picchio plan: usage: picchio.py plan [MODEL]")
    mach = machine_info()
    budget, blabel = plan_budget(mach)
    bw, speed_note = plan_speed_source(load_cache())
    if argv:
        name, fb, meta, note = plan_target(argv[0])
        row = plan_row(name, fb, meta, note, budget, bw)
        print(colorize(render_plan_one(row, budget, blabel, bw,
                                       speed_note)))
        sys.exit(0)
    sizes = {}
    if ollama_reachable():
        try:
            for m in ollama_api("/api/tags", timeout=5).get("models", []):
                if m.get("name"):
                    sizes[m["name"]] = m.get("size")
        except (urllib.error.URLError, OSError, ValueError):
            pass
    rows = []
    for label, note, arg, _size in scan_models()[0]:
        if note == "gguf":
            n, fb, meta, why = plan_target(arg)
            rows.append(plan_row(n, fb, meta, why, budget, bw))
        elif note == "ollama":
            n, fb, meta, why = plan_target(arg)
            fb = fb or sizes.get(arg)
            rows.append(plan_row(n, fb, meta, why, budget, bw))
        else:
            rows.append({"name": label, "need": None, "est": None,
                         "moe": False, "state": "not judged",
                         "note": note})
    if not rows:
        sys.exit("picchio plan: no models found on this machine; give "
                 "it a .gguf path or an ollama tag.")
    print(colorize(render_plan_scan(rows, budget, blabel, bw, speed_note)))
    sys.exit(0)


# --------------------------------------------------- id (effective identity)
#
# picchio id splits the one word people trade ("4bit") back into the
# three axes it actually is. The weight recipe: general.file_type is a
# recipe name, and every recipe mixes per-tensor types, so the card
# walks the tensor table and prices each tensor by its ggml type into
# one effective bits-per-weight figure. The kv cache: a runtime flag,
# never in the file, so the card cites only a dtype this machine has
# measured. The experts: on a mixture of experts most weights sit
# parked, and expert_used_count over expert_count is how many wake per
# token. Same contract as plan: read only, no verdict block, exit 0.

# ggml type number -> (name, bytes per block, elements per block),
# probed from this machine's own libggml 0.13.1 (the library the
# b9430 binaries link) through ggml_type_name / ggml_type_size /
# ggml_blck_size. Removed and deprecated slots are absent on purpose:
# an unknown number refuses loudly instead of guessing a size.
GGML_TENSOR_TYPES = {
    0: ("f32", 4, 1), 1: ("f16", 2, 1), 2: ("q4_0", 18, 32),
    3: ("q4_1", 20, 32), 6: ("q5_0", 22, 32), 7: ("q5_1", 24, 32),
    8: ("q8_0", 34, 32), 9: ("q8_1", 36, 32), 10: ("q2_K", 84, 256),
    11: ("q3_K", 110, 256), 12: ("q4_K", 144, 256),
    13: ("q5_K", 176, 256), 14: ("q6_K", 210, 256),
    15: ("q8_K", 292, 256), 16: ("iq2_xxs", 66, 256),
    17: ("iq2_xs", 74, 256), 18: ("iq3_xxs", 98, 256),
    19: ("iq1_s", 50, 256), 20: ("iq4_nl", 18, 32),
    21: ("iq3_s", 110, 256), 22: ("iq2_s", 82, 256),
    23: ("iq4_xs", 136, 256), 24: ("i8", 1, 1), 25: ("i16", 2, 1),
    26: ("i32", 4, 1), 27: ("i64", 8, 1), 28: ("f64", 8, 1),
    29: ("iq1_m", 56, 256), 30: ("bf16", 2, 1), 34: ("tq1_0", 54, 256),
    35: ("tq2_0", 66, 256), 39: ("mxfp4", 17, 32),
    40: ("nvfp4", 36, 64), 41: ("q1_0", 18, 128),
}

# general.file_type number -> recipe name, from the llama_ftype enum
# in this machine's b9430 llama.h (removed slots absent, same rule)
LLAMA_FTYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS",
    31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}


def gguf_tensor_table(f, n_tensors):
    """The descriptor table between the kv section and the tensor
    data, layout verified against the local 9B file byte by byte: per
    tensor a u64-length name, u32 dimension count, u64 dims fastest
    first, u32 ggml type, u64 offset relative to the aligned start of
    the data section. Returns (descriptors, header end position)."""
    if n_tensors > 65536:
        raise ValueError("gguf header claims {} tensors".format(n_tensors))
    out = []
    for _ in range(n_tensors):
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 1 << 16:
            raise ValueError("gguf tensor name of {} bytes".format(n))
        name = f.read(n).decode("utf-8", "replace")
        nd = struct.unpack("<I", f.read(4))[0]
        if nd > 8:
            raise ValueError("{} claims {} dimensions".format(name, nd))
        dims = struct.unpack("<{}Q".format(nd), f.read(8 * nd))
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        out.append((name, dims, ttype, off))
    return out, f.tell()


def ollama_tensor_table(ts):
    """/api/show tensors (name, type string, shape) mapped onto the
    same descriptor tuples the file walk yields. The api mirrors the
    table without offsets (measured on 0.31.1), so only the type
    arithmetic can price it; the offset audit is file-only."""
    byname = {v[0].lower(): k for k, v in GGML_TENSOR_TYPES.items()}
    out = []
    for t in ts:
        tt = byname.get(str(t.get("type", "")).lower())
        if tt is None:
            raise ValueError("unknown tensor type {!r} on {}".format(
                t.get("type"), t.get("name", "?")))
        out.append((t.get("name", "?"),
                    tuple(int(d) for d in t.get("shape", [])), tt, None))
    return out


def id_account(tensors, data_bytes=None, align=32):
    """({type name: [tensors, elements, bytes]}, elements, bytes),
    priced two ways when the file is at hand. Method one is type
    arithmetic: elements over the block size times the block bytes.
    Method two is the header's own offsets: each tensor must end
    within one alignment unit of the next offset, the last within one
    unit of the data section end. Both landed on the same byte total
    on both local files (zero padding); a mismatch raises, because a
    wrong triple or a misread table must never print a number."""
    hist, elems, total, priced = {}, 0, 0, []
    for name, dims, tt, off in tensors:
        if tt not in GGML_TENSOR_TYPES:
            raise ValueError("unknown ggml type {} on {}".format(tt, name))
        tname, tsize, blck = GGML_TENSOR_TYPES[tt]
        n = 1
        for d in dims:
            n *= int(d)
        if not n or n % blck:
            raise ValueError("{} elements do not fill {} blocks"
                             .format(name, tname))
        b = n // blck * tsize
        h = hist.setdefault(tname, [0, 0, 0])
        h[0] += 1
        h[1] += n
        h[2] += b
        elems += n
        total += b
        priced.append((name, off, b))
    if not elems:
        raise ValueError("the tensor table is empty")
    if data_bytes is not None:
        priced.sort(key=lambda t: t[1])
        for i, (name, off, b) in enumerate(priced):
            nxt = priced[i + 1][1] if i + 1 < len(priced) else data_bytes
            if not (off + b <= nxt < off + b + align):
                raise ValueError("offset audit failed at {}: the typed "
                                 "size does not meet the next offset"
                                 .format(name))
    return hist, elems, total


def id_experts(meta, tensors, elems):
    """(used, count, active elements) or None on a dense model. An
    expert bank is any tensor whose slowest dimension equals
    expert_count: on the local 35B that selects exactly the
    ffn_{down,gate,up}_exps banks, and the api mirror reports the
    same dimension order, so one rule serves both sources."""
    count = _arch_get(meta, "expert_count")
    if not count:
        return None
    used = int(_arch_get(meta, "expert_used_count") or 0)
    bank = 0
    for name, dims, tt, off in tensors:
        if len(dims) >= 3 and int(dims[-1]) == int(count):
            n = 1
            for d in dims:
                n *= int(d)
            bank += n
    active = elems - bank + bank * used // int(count)
    return used, int(count), active


def id_claim(recipe, name):
    """What the model says it is before any walking: the declared
    recipe name against the quant token the file or tag name carries.
    Both are claims; the table walk is what checks them."""
    m = re.findall(r"(?i)\b(?:[it]?q\d[0-9a-z_]*|bf16|f16|f32|mxfp4|"
                   r"nvfp4)\b", name)
    token = max(m, key=len) if m else None
    if recipe and token:
        if token.upper() == recipe.upper():
            return "{} (general.file_type; the name agrees)".format(recipe)
        return "{} in general.file_type, but the name says {}".format(
            recipe, token)
    if recipe:
        return "{} (general.file_type; no quant token in the name)" \
            .format(recipe)
    if token:
        return "{} from the name only; no general.file_type".format(token)
    return "none: no general.file_type, no quant token in the name"


def id_kv_note(cache):
    """The kv axis only ever cites a dtype this machine has measured:
    a run's stderr K/V markers land in the cache, and with none on
    file the card says not measured instead of assuming f16."""
    kt = (cache or {}).get("kv_types")
    if kt:
        return ("a runtime choice, not in the model. K {}, V {} on the "
                "last measured run here ({}, {}); -ctk / -ctv move it "
                "per run".format(kt[0], kt[1],
                                 cache.get("model_name", "?"),
                                 str(cache.get("stamp", "?"))[:10]))
    return ("a runtime choice, not in the model, and no measured run "
            "on this machine has recorded it yet. Measure once "
            "(python3 picchio.py MODEL) and the card cites that run; "
            "-ctk / -ctv move it per run")


def _id_wrap(label, text):
    return textwrap.wrap(text, width=WIDTH,
                         initial_indent="  " + label.ljust(11),
                         subsequent_indent=" " * 13)


def render_id(name, claim, acct, moe, kv_note, audit_note):
    """The identity card: claim, walked mixture, effective bits per
    weight, then the axes the file cannot carry. Information card
    contract, same as plan: no 15 line block, no mp1 footer."""
    out = ["picchio id: " + name]
    out += _id_wrap("claimed", claim)
    if not acct:
        out += _id_wrap("walked", "nothing: {}. The per tensor mix "
                        "lives in the table itself; point id at the "
                        ".gguf file for the walk.".format(audit_note))
        out += _id_wrap("kv cache", kv_note)
        return "\n".join(out)
    hist, elems, total = acct
    out += _id_wrap("walked", "{} tensors, {} types, priced one by "
                    "one:".format(sum(h[0] for h in hist.values()),
                                  len(hist)))
    for tname in sorted(hist, key=lambda k: -hist[k][2]):
        c, n, b = hist[tname]
        out.append("    {:<8}{:>5} tensors {:>6.2f} bits {:>6.1f}% of "
                   "weight bytes".format(tname, c, b * 8.0 / n,
                                         100.0 * b / total))
    out += _id_wrap("effective", "{:.2f} bits per weight: {:,} tensor "
                    "bytes over {:,} weights; {}".format(
                        total * 8.0 / elems, total, elems, audit_note))
    out += _id_wrap("kv cache", kv_note)
    if moe:
        used, count, active = moe
        out += _id_wrap("experts", "{} of {} wake per token: about "
                        "{:.1f}B of the {:.1f}B weights are read for "
                        "any one token".format(used, count,
                                               active / 1e9,
                                               elems / 1e9))
    return "\n".join(out)


def id_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio.py id MODEL\n"
              "split the quant label into the three axes it hides: the\n"
              "per tensor type mix priced into one effective bits per\n"
              "weight figure (walked from the gguf tensor table, offsets\n"
              "audited), the kv cache dtype (a runtime choice, cited\n"
              "only from a run measured here), and how many experts\n"
              "wake per token on a mixture of experts. A .gguf path is\n"
              "walked directly, an ollama tag through the api's mirror\n"
              "of the same table. Read only, never a verdict.")
        sys.exit(0)
    if len(argv) != 1:
        sys.exit("picchio id: usage: picchio.py id MODEL (a .gguf path "
                 "or an ollama tag)")
    arg = argv[0]
    kv_note = id_kv_note(load_cache())
    if os.path.isfile(arg):
        name = os.path.basename(arg)
        try:
            with open(arg, "rb") as f:
                meta = gguf_meta_stream(f)
                tensors, hdr_end = gguf_tensor_table(
                    f, meta.get("__tensor_count", 0))
            align = int(meta.get("general.alignment") or 32)
            data_start = (hdr_end + align - 1) // align * align
            acct = id_account(tensors, os.path.getsize(arg) - data_start,
                              align)
        except (ValueError, struct.error, OSError) as e:
            sys.exit("picchio id: {}: {}".format(name, e))
        claim = id_claim(
            LLAMA_FTYPES.get(meta.get("general.file_type")), name)
        print(render_id(name, claim, acct,
                        id_experts(meta, tensors, acct[1]), kv_note,
                        "the header's own offsets audit to the same "
                        "byte total"))
        sys.exit(0)
    if not looks_like_tag(arg):
        sys.exit("picchio id: no such file: {}".format(arg))
    if not ollama_reachable():
        sys.exit("picchio id: {!r} looks like an ollama tag, but no "
                 "ollama answered at {}.".format(arg, OLLAMA_HOST))
    try:
        show = ollama_api("/api/show", {"model": arg}, timeout=15)
    except (urllib.error.URLError, OSError, ValueError):
        sys.exit("picchio id: ollama at {} does not know the model "
                 "{!r}.".format(OLLAMA_HOST, arg))
    mi = show.get("model_info") or {}
    recipe = LLAMA_FTYPES.get(mi.get("general.file_type")) \
        or (show.get("details") or {}).get("quantization_level")
    claim = id_claim(recipe, arg)
    try:
        if not show.get("tensors"):
            raise ValueError("this ollama api answered without a "
                             "tensors field")
        tensors = ollama_tensor_table(show["tensors"])
        acct = id_account(tensors)
    except ValueError as e:
        print(render_id(arg, claim, None, None, kv_note, str(e)))
        sys.exit(0)
    print(render_id(arg, claim, acct, id_experts(mi, tensors, acct[1]),
                    kv_note, "typed shapes from the api, which mirrors "
                    "the table without offsets, so no offset audit"))
    sys.exit(0)


# ---------------------------------------------------------------- selftest

def selftest():
    """Replays the raw engine logs committed under examples/raw through
    the same parser, aggregation and diagnosis used live, re-renders each
    block, and requires it to match the committed example line for line
    (footer excluded: it names the machine that ran the replay)."""
    here = os.path.dirname(os.path.abspath(__file__))
    rawroot = os.path.join(here, "examples", "raw")
    if not os.path.isdir(rawroot):
        sys.exit("picchio: no examples/raw next to picchio.py")
    fx_ok = fx_all = rp_ok = rp_all = 0
    for name in sorted(os.listdir(rawroot)):
        d = os.path.join(rawroot, name)
        if not os.path.isdir(d):
            continue
        passes, metas = [], []
        for i in range(1, 32):
            stderr_p = os.path.join(d, "pass{}.stderr.txt".format(i))
            resp_p = os.path.join(d, "pass{}.response.json".format(i))
            meta_p = os.path.join(d, "pass{}.meta.json".format(i))
            if not os.path.exists(meta_p):
                break
            meta = json.load(open(meta_p))
            metas.append(meta)
            fx_all += 1
            if os.path.exists(stderr_p):
                p = parse_stderr(open(stderr_p).read(), meta["wall_s"])
                mode = "llama.cpp"
            elif os.path.exists(resp_p) and meta.get("mode") == "server":
                p = map_server(json.load(open(resp_p)), meta["wall_s"])
                mode = "server"
            elif os.path.exists(resp_p):
                p = map_ollama(json.load(open(resp_p)), meta["wall_s"],
                               meta.get("ps"))
                mode = "ollama"
            else:
                break
            if p["prefill_toks"] and p["decode_toks"] and p["wallclock_toks"]:
                fx_ok += 1
            passes.append(p)
        if not passes:
            continue
        rp_all += 1
        txt_p = os.path.join(here, "examples", name + ".txt")
        want = open(txt_p).read().rstrip().splitlines()
        l1, l2 = passes[0]["load_ms"], passes[1]["load_ms"]
        cold_note = (l1 is not None and l2 is not None
                     and l1 < 2 * l2 + 500)
        tele = None  # raw dirs that predate the sampler have no curve
        tj = os.path.join(d, "telemetry.json")
        if os.path.exists(tj):
            tele = json.load(open(tj)).get("summary")
        rep = build_rep(passes)
        state, para = diagnose(passes[0], rep, mode, tele)
        extra = metas[0].get("extra_args", [])
        why = attribute_why(state, rep, mode, extra)
        got = render_verdict(
            machine_info(), metas[0].get("engine", "?"),
            metas[0].get("model_name", "?"), passes, state, para, mode,
            None, cold_note, why, metas[0].get("ctx", effective_ctx(extra)),
            extra, tele).splitlines()
        if got[:-1] == want[:-1]:
            rp_ok += 1
        else:
            for a, b in zip(want, got):
                if a != b:
                    print("  {} mismatch:\n    want: {}\n    got:  {}".format(
                        name, a, b))
                    break
    # compare: the two committed llama.cpp blocks are a natural pair
    cp_ok, cp_all = 0, 4
    ha = open(os.path.join(here, "examples", "healthy-metal.txt")).read()
    fb = open(os.path.join(here, "examples", "cpu-fallback.txt")).read()
    pa = parse_block("someone posted this:\n" + ha + "\nhope it helps")
    pb = parse_block(fb)
    if pa and pb and pa["ctx"] == CTX \
            and pb["args"] == "--device none -ngl 0":
        cp_ok += 1
    two = render_compare(("A", "B"), pa, pb)
    if "SUSPECT: placement" in two and "0/33" in two:
        cp_ok += 1
    if "Nothing to compare" in render_compare(("A", "A"), pa,
                                              parse_block(ha)):
        cp_ok += 1
    # old format: strip the two fingerprint fields and the committed
    # block is byte for byte the pre-fingerprint output (they are the
    # only format change since); it must parse as unknown, not a guess
    old = re.sub(r"(?m)^(ctx \d+)", lambda m: " " * len(m.group(1)), fb)
    po = parse_block(re.sub(r"(?m)^(gpu\s{6}.*) \[.*\]$", r"\1", old))
    if po and po["ctx"] is None and po["args"] is None \
            and po["frac"] == 0.0 \
            and "unknown" in render_compare(("A", "B"), pa, po):
        cp_ok += 1
    # telemetry: synthetic timelines pushed through the real window
    # math and the real three source judge (no gpu needed, ci safe)
    te_ok, te_all = 0, 5
    gib = 1024 ** 3

    def synth_tele(idle_dev, work_dev, mem_base, mem_peak, src=None):
        # one 11.2 s pass after a 1.2 s baseline, ticked at 4 Hz; busy
        # samples land exactly in the tail aligned compute window
        t_end, wall, load_s, prompt_s, eval_s = 12.4, 11.2, 2.0, 1.3, 6.3
        dec1 = t_end - TELE_PAD_S
        pre0 = dec1 - eval_s - prompt_s
        t0, samples, t = t_end - wall, [], 0.0
        while t < t_end + 0.5:
            in_work = pre0 <= t <= dec1
            samples.append({
                "t": t,
                "dev": work_dev if in_work else idle_dev,
                "gpu_w": 10.6 if in_work and work_dev >= 50 else 0.02,
                "mem": mem_peak if t >= t0 + load_s else mem_base})
            t += 0.25
        return telemetry_summary(samples, [{
            "t_end": t_end, "wall_s": wall, "load_s": load_s,
            "prompt_s": prompt_s, "eval_s": eval_s}], src=src)

    fx = blank_pass()
    fx.update(offload_n=33, offload_total=33, prefill_toks=558.9,
              decode_toks=20.0, model_bytes=int(5.28 * gib))
    busy = synth_tele(0, 99, 600 * 1024 ** 2, int(7.0 * gib))
    flat = synth_tele(0, 0, 600 * 1024 ** 2, 700 * 1024 ** 2)
    # 1: gpu busy aligned with the compute window backs the full claim
    if telemetry_vote(busy, fx, "llama.cpp") == "agree" \
            and diagnose(fx, fx, "llama.cpp", busy)[0] == "HEALTHY" \
            and "mem +6.4 GiB, 10.6 W" in os_line(busy):
        te_ok += 1
    # 2: a flat line under a full offload claim is a two source fight;
    #    the WHY ladder stays out (the block itself is the exhibit)
    st, para = diagnose(fx, fx, "llama.cpp", flat)
    if st == "CONFLICTING EVIDENCE" and "stay flat" in para \
            and attribute_why(st, fx, "llama.cpp", []) is None:
        te_ok += 1
    # 3: a busy desktop disqualifies whole-gpu numbers: abstain, say so
    lifted = synth_tele(47, 99, 600 * 1024 ** 2, int(7.0 * gib))
    if telemetry_vote(lifted, fx, "llama.cpp") == "abstain" \
            and diagnose(fx, fx, "llama.cpp", lifted)[0] == "HEALTHY" \
            and "not judged" in os_line(lifted):
        te_ok += 1
    # 4: the memory step vetoes the flat line contradiction
    stepped = synth_tele(0, 0, 600 * 1024 ** 2, int(6.4 * gib))
    if telemetry_vote(stepped, fx, "llama.cpp") == "abstain":
        te_ok += 1
    # 5: timing physics alone still catches a cpu shaped full claim
    slow = dict(fx, prefill_toks=28.3, decode_toks=12.0)
    st, para = diagnose(slow, slow, "llama.cpp")
    if st == "CONFLICTING EVIDENCE" and "CPU shaped" in para \
            and attribute_why(st, slow, "llama.cpp", []) is None:
        te_ok += 1
    # verify: the two committed blocks pass, and blocks tampered by one
    # edit fail. Fixtures are built in memory from the real examples, so
    # no forged block ships in the repo; ha and fb are read above.
    ve_ok, ve_all = 0, 4
    if verify_block(parse_block(ha))[0] == "PASS":
        ve_ok += 1
    if verify_block(parse_block(fb))[0] == "PASS":
        ve_ok += 1
    # flip the cpu-fallback block's placement line to claim the full gpu:
    # one edit, and three independent witnesses (the ratio, the os meter,
    # the headline) each catch the run's real cpu shape underneath
    forged = re.sub(r"gpu      NOT ENGAGED: 0/33 layers on GPU \[.*\]",
                    "gpu      ENGAGED: 33/33 layers on GPU (Metal: Apple M5)",
                    fb)
    fv, ff = verify_block(parse_block(forged))
    if fv == "FLAG" and any("cpu shaped" in x for x in ff) and len(ff) >= 3:
        ve_ok += 1
    # invert a lane so decode reads faster than prefill: pure physics,
    # impossible on one run, caught with no hardware knowledge at all
    inv = re.sub(r"(warm mid\s+)588\.0 tok/s(\s+)21\.1 tok/s",
                 r"\g<1>15.0 tok/s\g<2>21.1 tok/s", ha)
    iv, iff = verify_block(parse_block(inv))
    if iv == "FLAG" and any("outrun" in x for x in iff):
        ve_ok += 1
    # watch: synthetic sample windows pushed through the real summary and
    # the real machine level judge (no gpu needed, ci safe)
    wa_ok, wa_all = 0, 3

    def synth_watch(dev_seq, mem, watt):
        return [{"t": i * 0.25, "dev": d, "mem": mem, "gpu_w": watt}
                for i, d in enumerate(dev_seq)]

    # 1: a busy window reads BUSY and states the whole-gpu caveat
    sb, pb = watch_verdict(watch_summary(
        synth_watch([0, 98, 99, 97, 99, 98], int(6.5 * gib), 12.0)),
        "runner (pid 1)")
    if sb == "GPU BUSY" and "machine level" in pb:
        wa_ok += 1
    # 2: a flat window reads IDLE and names the cpu
    si, pi = watch_verdict(watch_summary(
        synth_watch([1, 2, 0, 3, 1, 2], 600 * 1024 ** 2, 0.03)),
        "runner (pid 1)")
    if si == "GPU IDLE" and "on the cpu" in pi:
        wa_ok += 1
    # 3: a middling window reads MIXED, not a false BUSY or IDLE
    sm, _ = watch_verdict(watch_summary(
        synth_watch([30, 35, 28, 40, 33], int(4.0 * gib), 6.0)), None)
    if sm == "GPU MIXED":
        wa_ok += 1
    # monitor: the per probe signature classifier and the session summary,
    # both pure, so ci needs no live server
    mo_ok, mo_all = 0, 4
    # a gpu shaped probe reads OK, a cpu shaped one FLAG (the same 5x/15x
    # lines the server block uses), a missing rate convicts nobody
    if monitor_classify(588.0, 21.1)[0] == "OK" \
            and monitor_classify(26.8, 12.2)[0] == "FLAG" \
            and monitor_classify(None, 21.0)[0] == "NODATA":
        mo_ok += 1
    # the unsure band is neither: a prefill slow but not cpu slow
    if monitor_classify(180.0, 20.0)[0] == "WATCH":
        mo_ok += 1
    # a session that flipped gpu->cpu->gpu counts two transitions, convicts
    # on the one cpu probe, and keeps that probe's ratio as the worst
    flap = monitor_summarize([("OK", 27.0), ("OK", 26.0), ("FLAG", 2.2),
                              ("OK", 25.0)])
    if flap["flag"] == 1 and flap["transitions"] == 2 \
            and abs(flap["worst_ratio"] - 2.2) < 1e-9:
        mo_ok += 1
    # an all healthy session names no fallback (the exit 0 shape)
    steady = monitor_summarize([("OK", 27.0), ("OK", 26.5), ("OK", 28.1)])
    if steady["flag"] == 0 and steady["transitions"] == 0 \
            and "ENGAGED throughout" in monitor_summary_line(steady):
        mo_ok += 1
    # ctx sweep: the slope sentence is exact on synthetic rows, and the
    # committed real sweep replays like a verdict block when present
    sw_ok, sw_all = 0, 2

    def row(ctx, depth, pf, dc, wc):
        return {"ctx": ctx, "depth": depth, "prefill": pf,
                "decode": dc, "wallclock": wc}

    decayed = [row(4096, 2800, 560.0, 21.0, 18.0),
               row(32768, 22000, 320.0, 15.0, 12.0)]
    if "decode fell 29%" in sweep_slope(decayed) \
            and "held within" in sweep_slope(
                [row(4096, 2800, 560.0, 21.0, 18.0),
                 row(32768, 22000, 500.0, 20.6, 17.0)]):
        sw_ok += 1
    swroot = os.path.join(here, "examples", "raw", "ctx-sweep")
    swtxt = os.path.join(here, "examples", "ctx-sweep.txt")
    if os.path.exists(os.path.join(swroot, "sweep.meta.json")) \
            and os.path.exists(swtxt):
        sm = json.load(open(os.path.join(swroot, "sweep.meta.json")))
        rows = []
        for ctx in sm["tiers"]:
            ps = []
            for i in range(1, sm["passes"] + 1):
                base = os.path.join(swroot, "ctx{}.pass{}".format(ctx, i))
                w = json.load(open(base + ".meta.json"))["wall_s"]
                if os.path.exists(base + ".stderr.txt"):
                    ps.append(parse_stderr(open(base + ".stderr.txt").read(), w))
                elif os.path.exists(base + ".response.json"):
                    ps.append(map_ollama(json.load(open(base + ".response.json")),
                                         w, None))
            if ps:
                rp = build_rep(ps)
                rows.append(row(ctx, rp.get("prompt_tokens"),
                                rp["prefill_toks"], rp["decode_toks"],
                                rp["wallclock_toks"]))
        got = render_sweep(machine_info(), sm["engine"],
                           sm["model_name"], rows).splitlines()
        want = open(swtxt).read().rstrip().splitlines()
        if got[:-1] == want[:-1]:  # footer names the replaying machine
            sw_ok += 1
    else:
        sw_all = 1  # no committed sweep yet: only the synthetic check runs
    # server endpoint judge: no engine claim exists over http, so the
    # two witnesses (os meter, speed signature) vote through the real
    # diagnose path; the synthetic telemetry timelines above are reused
    sv_ok, sv_all = 0, 4
    sfx = blank_pass()
    sfx.update(prefill_toks=560.0, decode_toks=20.0)  # 28x, gpu shaped
    st, para = diagnose(sfx, sfx, "server", busy)
    if st == "HEALTHY" and "os meter" in para and "gpu shaped" in para:
        sv_ok += 1
    cpu_fx = dict(sfx, prefill_toks=48.0, decode_toks=12.0)  # 4x, cpu
    st, para = diagnose(cpu_fx, cpu_fx, "server", flat)
    if st == "SILENT CPU FALLBACK" and "on the cpu" in para \
            and "server api" in attribute_why(st, cpu_fx, "server", []):
        sv_ok += 1
    st, para = diagnose(cpu_fx, cpu_fx, "server", busy)  # witnesses fight
    if st == "CONFLICTING EVIDENCE" and "Believe neither" in para:
        sv_ok += 1
    midr = dict(sfx, prefill_toks=200.0, decode_toks=20.0)  # 10x dead zone
    st, para = diagnose(midr, midr, "server", None)
    if st == "NO PLACEMENT EVIDENCE" and "placement is not" in para:
        sv_ok += 1
    # linux parser: the four graduated 4090 stderr shapes, each pinned
    # on the fields the diagnosis reads (all captured on b9430 CUDA and
    # cpu-only builds, driver 550.54.14)
    lx_ok, lx_all = 0, 4
    lxroot = os.path.join(here, "examples", "raw", "linux-4090")

    def lparse(fname):
        p = os.path.join(lxroot, fname)
        return parse_stderr(open(p).read(), None) if os.path.exists(p) \
            else None

    lx_h = lparse("cuda-healthy.stderr.txt")
    lx_m = lparse("misbuilt-cpu.stderr.txt")
    if lx_h and lx_h["offload_n"] == 33 and lx_h["offload_total"] == 33 \
            and lx_h["gpu_kind"] == "CUDA" \
            and lx_h["gpu_device"] == "GeForce RTX 4090" \
            and lx_h["free_mib"] == 23818:
        lx_ok += 1
    lx_z = lparse("cuda-ngl0.stderr.txt")
    if lx_z and lx_z["offload_n"] == 0 and lx_z["offload_total"] == 33 \
            and lx_z["gpu_kind"] == "CUDA":
        lx_ok += 1
    lx_p = lparse("cuda-partial.stderr.txt")
    if lx_p and lx_p["offload_n"] == 10 and lx_p["offload_total"] == 33:
        lx_ok += 1
    # the misbuilt build prints no offload line and no device line at
    # all; that absence is exactly what the silent-engine rule needs
    if lx_m and lx_m["offload_n"] is None and lx_m["gpu_kind"] is None \
            and lx_m["gpu_device"] is None and lx_m["threads"] == 48:
        lx_ok += 1
    # silent-engine: with no engine claim, an nvml flat line on an idle
    # machine convicts, and each of the five gates alone acquits
    se_ok, se_all = 0, 4
    se_fx = blank_pass()
    se_fx.update(prefill_toks=16.6, decode_toks=1.1,
                 model_bytes=int(5.28 * gib))
    se_flat = synth_tele(0, 0, 354 * 1024 ** 2, 354 * 1024 ** 2,
                         src="nvml")
    # 1: conviction, plus the memory step veto acquitting the same run
    st, para = diagnose(se_fx, se_fx, "llama.cpp", se_flat)
    se_step = synth_tele(0, 0, 354 * 1024 ** 2, int(6.0 * gib),
                         src="nvml")
    if st == "SILENT CPU FALLBACK" and "printed no gpu evidence" in para \
            and "stayed idle" in attribute_why(st, se_fx, "llama.cpp", []) \
            and diagnose(se_fx, se_fx, "llama.cpp",
                         se_step)[0] == "NO PLACEMENT EVIDENCE":
        se_ok += 1
    # 2: a busy desktop abstains, no conviction on a lifted baseline
    se_busy = synth_tele(47, 47, 354 * 1024 ** 2, 354 * 1024 ** 2,
                         src="nvml")
    if diagnose(se_fx, se_fx, "llama.cpp",
                se_busy)[0] == "NO PLACEMENT EVIDENCE":
        se_ok += 1
    # 3: no nvml, no upgrade: a cpu only machine keeps the old verdict,
    #    and so does the same flat line without the nvml source mark
    if diagnose(se_fx, se_fx, "llama.cpp",
                {"off": "no nvml"})[0] == "NO PLACEMENT EVIDENCE" \
            and diagnose(se_fx, se_fx, "llama.cpp",
                         synth_tele(0, 0, 1, 1))[0] \
            == "NO PLACEMENT EVIDENCE":
        se_ok += 1
    # 4: any engine claim keeps the old path: 0/33 with a flat curve is
    #    the ordinary fallback with the ladder WHY, not the silent one
    se_cl = dict(se_fx, offload_n=0, offload_total=33)
    st, para = diagnose(se_cl, se_cl, "llama.cpp", se_flat)
    if st == "SILENT CPU FALLBACK" and "printed no gpu evidence" not in para \
            and "engine log does not say" in attribute_why(
                st, se_cl, "llama.cpp", []):
        se_ok += 1
    # real curve regression: the two 4090 telemetry captures replay
    # through the real window math; the misbuilt one must convict and
    # the healthy one must not
    rc_ok, rc_all = 0, 2

    def load_curve(name):
        rows = []
        path = os.path.join(lxroot, name + ".telemetry.jsonl")
        if not os.path.exists(path):
            return rows
        for line in open(path):
            d = json.loads(line)
            if "util_gpu" in d:
                rows.append({"t": d["t"], "dev": d["util_gpu"],
                             "mem": d["mem_used_mib"] * 1024 ** 2,
                             "gpu_w": d.get("power_w")})
        return rows

    def curve_marks(samples, meta_name, rep):
        meta = json.load(open(os.path.join(lxroot, meta_name)))
        return [{"t_end": samples[-1]["t"], "wall_s": meta["wall_s"],
                 "load_s": (rep["load_ms"] or 0) / 1000.0,
                 "prompt_s": (rep["prompt_ms"] or 0) / 1000.0,
                 "eval_s": (rep["eval_ms"] or 0) / 1000.0}]

    rc_m = load_curve("misbuilt-cpu")
    if rc_m and lx_m:
        mtele = telemetry_summary(
            rc_m, curve_marks(rc_m, "misbuilt-cpu.meta.json", lx_m),
            src="nvml")
        if telemetry_read(mtele) == "flat" \
                and diagnose(lx_m, lx_m, "llama.cpp",
                             mtele)[0] == "SILENT CPU FALLBACK":
            rc_ok += 1
    rc_h = load_curve("cuda-healthy")
    if rc_h and lx_h:
        htele = telemetry_summary(
            rc_h, curve_marks(rc_h, "cuda-healthy.meta.json", lx_h),
            src="nvml")
        if diagnose(lx_h, lx_h, "llama.cpp", htele)[0] == "HEALTHY" \
                and telemetry_vote(htele, lx_h,
                                   "llama.cpp") in ("agree", "abstain"):
            rc_ok += 1
    # plan: a synthetic gguf header replays through the real reader,
    # and the kv formula must land on the engine's own committed
    # allocation figures; the speed gate refuses everything but a
    # cached dense measurement
    pl_ok, pl_all = 0, 6

    def synth_gguf(arch, kvs):
        out = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", 0),
               struct.pack("<Q", len(kvs) + 1)]

        def emit(key, t, packed):
            out.append(struct.pack("<Q", len(key)) + key.encode())
            out.append(struct.pack("<I", t) + packed)

        emit("general.architecture", 8,
             struct.pack("<Q", len(arch)) + arch.encode())
        for k, v in kvs.items():
            emit(arch + "." + k, 4, struct.pack("<I", v))
        return io.BytesIO(b"".join(out))

    m9 = gguf_meta_stream(synth_gguf("qwen35", {
        "block_count": 32, "attention.head_count": 16,
        "attention.head_count_kv": 4, "attention.key_length": 256,
        "attention.value_length": 256, "full_attention_interval": 4}))
    kv9, note9 = kv_account(m9)
    # 1: reader plus formula reproduce the engine's own 128.00 MiB
    #    llama_kv_cache line (examples/raw/healthy-metal, ctx 4096)
    if m9["qwen35.block_count"] == 32 and kv9 == 128 * 1024 ** 2 \
            and "8 of 32" in note9:
        pl_ok += 1
    m35 = gguf_meta_stream(synth_gguf("qwen35moe", {
        "block_count": 40, "attention.head_count": 16,
        "attention.head_count_kv": 2, "attention.key_length": 256,
        "attention.value_length": 256, "full_attention_interval": 4,
        "expert_count": 256}))
    kv35, _n = kv_account(m35)
    # 2: the moe kv comes from the attention geometry alone: 80.00 MiB,
    #    matching the engine's allocation for the local 35B
    if kv35 == 80 * 1024 ** 2 and plan_is_moe(m35) and not plan_is_moe(m9):
        pl_ok += 1
    # 3: head_dim falls back to embedding over heads when the header
    #    has no key/value length (the classic dense layout)
    kvf, _n = kv_account(gguf_meta_stream(synth_gguf("llama", {
        "block_count": 32, "attention.head_count": 32,
        "attention.head_count_kv": 8, "embedding_length": 4096})))
    if kvf == 4096 * 32 * 8 * 256 * 2:
        pl_ok += 1
    # 4: the fit bands sit where the calibration put them (the 35B ran
    #    healthy at 85% of budget, so 80% fits, 100% tight, 112% no)
    if plan_state(20 * gib, 25 * gib) == "fits" \
            and plan_state(25 * gib, 25 * gib) == "tight" \
            and plan_state(28 * gib, 25 * gib) == "no":
        pl_ok += 1
    # 5: no cached run, no speed: the refusal is explicit, not a guess
    bw, note = plan_speed_source(None)
    if bw is None and "not calibrated" in note:
        pl_ok += 1
    # 6: a cached dense run prices a dense target and refuses a moe
    #    target; a cached moe run refuses to calibrate at all
    bw, note = plan_speed_source({"model_bytes": 5 * gib, "moe": False,
                                  "model_name": "m",
                                  "rates": {"decode": 20.0}})
    mbw, mnote = plan_speed_source({"model_bytes": 5 * gib, "moe": True,
                                    "model_name": "m",
                                    "rates": {"decode": 20.0}})
    if bw == 100 * gib and plan_est_decode(bw, 10 * gib, False) == 10.0 \
            and plan_est_decode(bw, 10 * gib, True) is None \
            and mbw is None and "mixture of experts" in mnote:
        pl_ok += 1
    # id: a synthetic gguf with a real tensor table replays through the
    # same walk, account and expert arithmetic used live (the big real
    # files stay out of ci; they are the manual acceptance step). The
    # engine side of the cross check reads the committed real stderr.
    id_ok, id_all = 0, 6

    def synth_id_img(specs, kvs):
        # a minimal legal gguf v3 image: kv section, tensor table,
        # data section sized and aligned exactly like the real files
        arch = "synthmoe"
        pairs = [("general.architecture", 8,
                  struct.pack("<Q", len(arch)) + arch.encode()),
                 ("general.file_type", 4, struct.pack("<I", 15))]
        pairs += [(arch + "." + k, 4, struct.pack("<I", v))
                  for k, v in kvs]
        kvb = b"".join(struct.pack("<Q", len(k)) + k.encode()
                       + struct.pack("<I", t) + p for k, t, p in pairs)
        off, rows = 0, []
        for name, dims, tt in specs:
            _tn, tsize, blck = GGML_TENSOR_TYPES[tt]
            n = 1
            for d in dims:
                n *= d
            rows.append(struct.pack("<Q", len(name)) + name.encode()
                        + struct.pack("<I", len(dims))
                        + struct.pack("<{}Q".format(len(dims)), *dims)
                        + struct.pack("<I", tt) + struct.pack("<Q", off))
            off += (n // blck * tsize + 31) // 32 * 32
        img = (b"GGUF" + struct.pack("<I", 3)
               + struct.pack("<Q", len(specs))
               + struct.pack("<Q", len(pairs)) + kvb + b"".join(rows))
        return img + b"\0" * (-len(img) % 32) + b"\0" * off

    img = synth_id_img([("blk.0.attn_q.weight", (256, 4), 12),
                        ("blk.0.ffn_down_exps.weight", (256, 2, 4), 13),
                        ("output_norm.weight", (256,), 0)],
                       [("expert_count", 4), ("expert_used_count", 2)])
    fh = io.BytesIO(img)
    idm = gguf_meta_stream(fh)
    idt, id_hdr_end = gguf_tensor_table(fh, idm.get("__tensor_count", 0))
    id_data = len(img) - (id_hdr_end + 31) // 32 * 32
    idh, ide, idb = id_account(idt, id_data)
    # 1: the walk reads the claim and the table, and the two pricing
    #    methods close. Priced by hand from the machine's own libggml
    #    triples: 1024 q4_K elements are 576 bytes (144 per 256
    #    block), 2048 q5_K are 1408, 256 f32 are 1024.
    if LLAMA_FTYPES.get(idm.get("general.file_type")) == "Q4_K_M" \
            and idh["q4_K"] == [1, 1024, 576] \
            and idh["q5_K"] == [1, 2048, 1408] \
            and idh["f32"] == [1, 256, 1024] and (ide, idb) == (3328, 3008):
        id_ok += 1
    # 2: a lying offset and a type outside the pinned triples both
    #    refuse to price instead of printing a wrong number
    ok2 = 0
    try:
        id_account([idt[0], (idt[1][0], idt[1][1], idt[1][2],
                             idt[1][3] + 64), idt[2]], id_data)
    except ValueError:
        ok2 += 1
    try:
        id_account([("x", (32,), 4, 0)])
    except ValueError:
        ok2 += 1
    if ok2 == 2:
        id_ok += 1
    # 3: the engine's own loader census crosses the walk: the committed
    #    healthy fixture reports the same five type counts the real
    #    file's table measured (verified against the file the day this
    #    landed), and its kv marker parses to f16/f16
    hp = parse_stderr(open(os.path.join(
        rawroot, "healthy-metal", "pass1.stderr.txt")).read(), 10.0)
    if hp["tensor_types"] == {"f32": 177, "q8_0": 48, "q4_K": 132,
                              "q5_K": 48, "q6_K": 22} \
            and hp["kv_types"] == ["f16", "f16"]:
        id_ok += 1
    # 4: the non-f16 sample measured here (-ctk q8_0 -ctv q8_0,
    #    committed raw) pins the K (q8_0) line shape
    qp = parse_stderr(open(os.path.join(
        rawroot, "kv-q8", "ctk-q8.stderr.txt")).read(), 1.0)
    if qp["kv_types"] == ["q8_0", "q8_0"]:
        id_ok += 1
    # 5: the expert bank is the slowest dimension matching
    #    expert_count: 2048 of 3328 elements park in banks, 2 of 4
    #    experts wake, so one token reads 2304; a dense header (no
    #    expert_count) renders no axis at all
    if id_experts(idm, idt, ide) == (2, 4, 2304) \
            and id_experts({"general.architecture": "d"}, idt, 9) is None:
        id_ok += 1
    # 6: the api mirror prices through the same account (no offsets to
    #    audit over http) and an unknown type string refuses
    ok6 = False
    try:
        oh, oe, ob = id_account(ollama_tensor_table(
            [{"name": "w", "type": "Q4_K", "shape": [256, 4]}]))
        ok6 = oh["q4_K"] == [1, 1024, 576] and (oe, ob) == (1024, 576)
        ollama_tensor_table([{"name": "w", "type": "Q9_Z",
                              "shape": [1]}])
        ok6 = False
    except ValueError:
        pass
    if ok6:
        id_ok += 1
    # argv split: the `--` passthrough is cut by hand before argparse,
    # so its semantics cannot vary with the interpreter (3.9.6 and
    # 3.12.3 rejected an option followed by `--` as unrecognized
    # arguments, measured; 3.12.13 and 3.13+ accept it). The command
    # shapes users actually type:
    av_ok, av_all = 0, 4
    # 1: options before the separator stay with picchio, engine args
    #    after it arrive verbatim (the exact shape old argparse rejects)
    if split_engine_args(["m.gguf", "--keep-logs", "d", "--", "-ngl", "0"]) \
            == (["m.gguf", "--keep-logs", "d"], ["-ngl", "0"]):
        av_ok += 1
    # 2: no separator, trailing junk stays put for the unexpected
    #    extra arguments error, never silently swallowed as engine args
    if split_engine_args(["m.gguf", "foo", "bar"]) \
            == (["m.gguf", "foo", "bar"], None):
        av_ok += 1
    # 3: no separator at all: nothing moves, None says none was typed
    if split_engine_args(["m.gguf"]) == (["m.gguf"], None):
        av_ok += 1
    # 4: only the first separator splits; any later one belongs to the
    #    engine command line untouched
    if split_engine_args(["m.gguf", "--", "-a", "--", "-b"]) \
            == (["m.gguf"], ["-a", "--", "-b"]):
        av_ok += 1
    # onboarding: the zero-argument entry decision is pure given what the
    # scan found, whether a terminal is attached, and what gets typed. The
    # four paths plus the two edges, none of them touching a tty or a gpu
    gd_ok, gd_all = 0, 8
    two = [("qwen3.5:9b", "ollama", "qwen3.5:9b", "5.3 GiB"),
           ("llama-3-8b.gguf", "gguf", "/models/llama-3-8b.gguf",
            "4.6 GiB")]
    one = [two[0]]

    def scripted(lines):
        it = iter(lines)
        return lambda prompt: next(it, None)

    def sink():
        out = []
        return out, out.append

    # 1: exactly one model on a terminal runs with no question asked
    log, emit = sink()
    if resolve_direction(one, True, scripted([]), emit) \
            == ("run", "qwen3.5:9b") \
            and any("Selected: qwen3.5:9b" in x for x in log):
        gd_ok += 1
    # 2: a real fork, the user types the menu number, that model runs
    log, emit = sink()
    if resolve_direction(two, True, scripted(["2"]), emit) \
            == ("run", "/models/llama-3-8b.gguf") \
            and "2 models found." in log:
        gd_ok += 1
    # 3: not a terminal (pipe/redirect) falls back to pasteable commands
    if resolve_direction(two, False, scripted([]), lambda s: None) \
            == ("print", None):
        gd_ok += 1
    # 4: the scan missed it, a typed path overrides the menu and runs
    if resolve_direction(two, True, scripted(["/tmp/my.gguf"]),
                         lambda s: None) == ("run", "/tmp/my.gguf"):
        gd_ok += 1
    # 5: nothing found but a terminal is on, the one prompt takes a tag
    log, emit = sink()
    if resolve_direction([], True, scripted(["some-tag:latest"]), emit) \
            == ("run", "some-tag:latest") and "No models found." in log:
        gd_ok += 1
    # 6: an out-of-range number re-asks, it never runs model zero
    log, emit = sink()
    if resolve_direction(two, True, scripted(["9", "1"]), emit) \
            == ("run", "qwen3.5:9b") \
            and any("No model 9" in x for x in log):
        gd_ok += 1
    # 7: a name longer than the column truncates in the menu row only;
    # the size column survives and the untouched full path still runs
    log, emit = sink()
    longlab = "L" * 47 + "-Q4_K_M.gguf"
    if resolve_direction(two + [(longlab, "gguf", "/m/" + longlab,
                                 "8.0 GiB")], True, scripted(["3"]),
                         emit) == ("run", "/m/" + longlab) \
            and any("..." in x and "8.0 GiB" in x for x in log):
        gd_ok += 1
    # 8: models found past the display cap are surfaced, not dropped in
    # silence; the header counts the true total and a trailer names how
    # many are hidden, and a typed number still reaches a shown one
    log, emit = sink()
    if resolve_direction(two, True, scripted(["1"]), emit, 5) \
            == ("run", "qwen3.5:9b") \
            and "7 models found." in log \
            and any("and 5 more" in x for x in log):
        gd_ok += 1
    vp_ok, vp_all = 0, 3
    if parse_engine_version("version: 9430 (d48a56ef)") == "b9430":
        vp_ok += 1
    # the tarball sentinel that once rendered as a fake build b0
    if parse_engine_version("version: 0 (unknown)") == "(version unknown)":
        vp_ok += 1
    if parse_engine_version("") == "(version unknown)":
        vp_ok += 1
    print("parser fixtures {}/{}, verdict replay {}/{}, compare {}/{}, "
          "telemetry {}/{}, verify {}/{}, watch {}/{}, monitor {}/{}, "
          "sweep {}/{}, server {}/{}, linux {}/{}, silent-engine {}/{}, "
          "curves {}/{}, plan {}/{}, id {}/{}, argv {}/{}, version {}/{}, "
          "onboarding {}/{}".format(
              fx_ok, fx_all, rp_ok, rp_all, cp_ok, cp_all, te_ok, te_all,
              ve_ok, ve_all, wa_ok, wa_all, mo_ok, mo_all, sw_ok, sw_all,
              sv_ok, sv_all, lx_ok, lx_all, se_ok, se_all, rc_ok, rc_all,
              pl_ok, pl_all, id_ok, id_all, av_ok, av_all, vp_ok, vp_all,
              gd_ok, gd_all))
    sys.exit(0 if fx_ok == fx_all and rp_ok == rp_all and rp_all
             and cp_ok == cp_all and te_ok == te_all
             and ve_ok == ve_all and wa_ok == wa_all and mo_ok == mo_all
             and sw_ok == sw_all and sv_ok == sv_all
             and lx_ok == lx_all and se_ok == se_all and rc_ok == rc_all
             and pl_ok == pl_all and id_ok == id_all and av_ok == av_all
             and vp_ok == vp_all and gd_ok == gd_all else 1)


# -------------------------------------------------------------------- main

def split_engine_args(argv):
    """Everything after the first bare `--` goes to the engine
    verbatim, and argparse never sees the separator. Splitting by hand
    is what keeps the passthrough identical on every Python: whether
    argparse itself honors a `--` that follows an option flipped
    between CPython versions (measured here: 3.9.6 and 3.12.3 reject
    "MODEL --keep-logs D -- -ngl 0" as unrecognized arguments, 3.12.13
    and 3.13+ accept it). Returns (argv_before, engine_args), with
    engine_args None when no separator was typed at all."""
    if "--" not in argv:
        return argv, None
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def save_cache(payload):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=1)
    except OSError:
        pass


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    # guard wraps an arbitrary user command, so its arguments must not
    # pass through the measurement mode parser: dispatch on the word
    if sys.argv[1:2] == ["guard"]:
        guard_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["compare"]:
        compare_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["verify"]:
        verify_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["watch"]:
        watch_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["monitor"]:
        monitor_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["plan"]:
        plan_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["id"]:
        id_cli(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(
        prog="picchio",
        description="Knocks on your local LLM setup and listens for hollow "
                    "spots: are your tok/s numbers what you think they are, "
                    "and did the GPU actually do the work?",
        epilog=(
            "glossary:\n"
            "  prefill    the model reading your prompt "
            "(prompt tokens per second)\n"
            "  decode     the model writing the answer "
            "(generated tokens per second)\n"
            "  wallclock  generated tokens over total elapsed time, "
            "load included\n"
            "  TTFT       time to first token, roughly load plus prefill "
            "when cold\n"
            "  offload    how many model layers sit on the GPU "
            "(0/33 = CPU run)\n"
            "\n"
            "server endpoint:\n"
            "  picchio.py http://127.0.0.1:8080\n"
            "  measure a llama-server you already have running, over its\n"
            "  own http api; no cold pass (the server stays loaded), and\n"
            "  placement is judged by the os meter and the speed signature\n"
            "\n"
            "guard mode:\n"
            "  picchio.py guard [--keep-logs DIR] -- <command...>\n"
            "  wrap your own llama.cpp command (llama-server, llama-cli);\n"
            "  warn the moment placement evidence shows layers off the\n"
            "  GPU, never kill it, summarize placement when it exits\n"
            "\n"
            "compare mode:\n"
            "  picchio.py compare A.txt B.txt\n"
            "  diff two pasted verdict blocks variable by variable and\n"
            "  name the first config difference that explains the gap\n"
            "\n"
            "verify mode:\n"
            "  picchio.py verify [FILE]\n"
            "  re-derive the physics a pasted verdict block claims and\n"
            "  flag it when placement, the speed signature, the os meter\n"
            "  and the headline do not describe the same run\n"
            "\n"
            "watch mode:\n"
            "  picchio.py watch [PID] [--engine ollama] [--for SEC]\n"
            "  point the os gpu meter at a running process or the whole\n"
            "  gpu and report whether the gpu is doing the work, without\n"
            "  parsing any engine's output (works for mlx, lm studio, ...)\n"
            "\n"
            "monitor mode:\n"
            "  picchio.py monitor URL [--every SEC] [--for SEC]\n"
            "  probe a running llama-server on a timer and flag any probe\n"
            "  whose prefill/decode signature goes cpu shaped: the\n"
            "  intermittent fallback a single snapshot cannot catch\n"
            "\n"
            "ctx sweep:\n"
            "  picchio.py model.gguf --ctx-sweep [4096,16384,32768]\n"
            "  re-measure the three lanes at each context depth, each one\n"
            "  filled for real, and report how far decode decays with depth\n"
            "\n"
            "plan mode:\n"
            "  picchio.py plan [MODEL]\n"
            "  the capacity account before you download or load: will it\n"
            "  fit, from the gguf header against this machine's memory\n"
            "  budget, plus an estimated decode rate once one diagnosis\n"
            "  has been measured here (estimates are always labeled)\n"
            "\n"
            "id mode:\n"
            "  picchio.py id MODEL\n"
            "  what the quant label actually holds: the per tensor type\n"
            "  mix priced into effective bits per weight, the kv cache\n"
            "  dtype (runtime, cited only from a run measured here), and\n"
            "  how many experts wake per token on a mixture of experts\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", nargs="?",
                    help="path to a .gguf file, an ollama model tag, or "
                         "the url of a running llama-server "
                         "(http://host:port)")
    ap.add_argument("--version", action="version",
                    version="picchio {} (protocol {})".format(
                        VERSION, PROTOCOL))
    ap.add_argument("--bin", help="llama.cpp binary (default: find "
                                  "llama-completion or llama-cli on PATH)")
    ap.add_argument("--passes", type=int, default=3, metavar="N",
                    help="measurement passes; the first is the cold one, "
                         "the verdict reports the warm median and span "
                         "(default 3, min 2)")
    ap.add_argument("--explain", type=float, metavar="TOKS",
                    help="classify a tok/s number you saw somewhere against "
                         "this machine's measured rates")
    ap.add_argument("--json", action="store_true",
                    help="print raw measurements as JSON after the verdict")
    ap.add_argument("--keep-logs", metavar="DIR",
                    help="save the raw engine output of each pass into DIR "
                         "(the evidence behind the verdict)")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="skip the OS-side GPU sampling; the os line then "
                         "says the verdict rests on engine+timing only")
    ap.add_argument("--ctx-sweep", nargs="?", const="4096,16384,32768",
                    metavar="LIST", dest="ctx_sweep",
                    help="re-measure the three lanes at each context size in "
                         "LIST (default 4096,16384,32768), each filled to "
                         "real depth, and report the decode decay slope")
    ap.add_argument("--selftest", action="store_true",
                    help="replay examples/raw through the parser and "
                         "diagnosis; verify the committed verdicts reproduce")
    ap.add_argument("extra", nargs="*", default=[],
                    help="args after -- go straight to the llama.cpp engine "
                         "(e.g. -- --device none -ngl 0)")
    argv, engine_args = split_engine_args(sys.argv[1:])
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return

    if args.model is None and args.explain is not None:
        cached = load_cache()
        if not cached:
            sys.exit("picchio: no previous run cached; run with a model "
                     "first.")
        verdict, para = classify_number(args.explain, cached["rates"])
        print(colorize("\n".join(
            ["YOUR NUMBER: {:.1f} tok/s -> {}".format(args.explain, verdict)]
            + wrap_para(para)
            + ["(rates: {}, {}, {})".format(
                cached.get("model_name", "?"), cached.get("machine", "?"),
                str(cached.get("stamp", "?"))[:10])])))
        return

    if args.model is None:
        # no direction from argv: the one place a model may be asked for.
        # a terminal on both ends means a person is watching; a pipe or a
        # redirect on either end stays composable and is never asked
        cands, dropped = scan_models()
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        action, chosen = resolve_direction(
            cands, interactive, _ask_line,
            lambda line: print(menu_paint(line)), dropped)
        if action == "run":
            args.model = chosen
        else:
            if action == "print":
                print_discovery(cands, dropped)
            elif not interactive:
                print(HINT_NO_MODELS)
            sys.exit(0)
    if args.passes < 2:
        sys.exit("picchio: --passes must be at least 2 (one cold, one warm).")
    if args.extra:
        # argparse never sees a `--` anymore, so whatever landed in the
        # extra positional is stray junk, not engine args
        sys.exit("picchio: unexpected extra arguments: {}\n"
                 "(a pasted trailing comment does this; engine args need "
                 "a bare -- first)".format(" ".join(args.extra)))
    args.extra = engine_args or []

    mach = machine_info()
    logdir = args.keep_logs
    lp = (lambda name: os.path.join(logdir, name)) if logdir else \
        (lambda name: None)

    mode, binpath, engine_str, model_name = resolve_engine(args.model,
                                                           args.bin)
    if mode != "llama.cpp" and args.extra:
        sys.exit("picchio: passthrough args after -- only work in "
                 "llama.cpp mode.")

    if args.ctx_sweep is not None:
        if mode == "server":
            sys.exit("picchio: --ctx-sweep sets the context size per "
                     "tier, and a server endpoint fixes it server side. "
                     "Run the sweep on the .gguf file instead.")
        # a separate diagnostic, not an mp1 verdict: it changes the prompt
        # per tier, so it prints its own block and never touches the cache
        rows = ctx_sweep(args.model, mode, binpath, engine_str, model_name,
                         parse_tiers(args.ctx_sweep), max(2, args.passes), lp)
        print(colorize(render_sweep(mach, engine_str, model_name, rows)))
        sys.exit(0)

    passes = []
    if mode == "ollama" and ollama_ps_entry(args.model):
        sys.stderr.write("picchio: unloading model for a colder pass 1 ...\n")
        ollama_unload(args.model)
    if mode == "server" and not url_is_local(binpath):
        # ioreg meters this machine; a remote server's gpu is not on it,
        # so the os witness recuses itself instead of testifying about
        # the wrong computer
        sampler = {"off": "remote endpoint", "ev": "timing"}
    else:
        sampler = telemetry_start(args.no_telemetry)
        if mode == "server" and isinstance(sampler, dict):
            sampler["ev"] = "timing"
    if isinstance(sampler, GpuSampler):
        time.sleep(1.2)  # a few ticks of idle baseline before pass 1
    block_ctx = server_ctx(binpath) if mode == "server" \
        else effective_ctx(args.extra)
    for i in range(args.passes):
        if i > 0:
            note = " (warm)"
        elif mode == "server":
            note = " (warm; the server is already loaded)"
        else:
            note = " (includes any cold load)"
        sys.stderr.write("picchio: pass {}{} ...\n".format(i + 1, note))
        if mode == "llama.cpp":
            p = run_llama_pass(binpath, args.model, args.extra,
                               lp("pass{}.stderr.txt".format(i + 1)))
            meta = {"wall_s": p["wall_s"], "engine": engine_str,
                    "model_name": model_name, "extra_args": args.extra}
        elif mode == "server":
            p = run_server_pass(
                binpath, lp("pass{}.response.json".format(i + 1)))
            meta = {"wall_s": p["wall_s"], "engine": engine_str,
                    "model_name": model_name, "mode": "server",
                    "ctx": block_ctx}
        else:
            p, ps = run_ollama_pass(
                args.model, lp("pass{}.response.json".format(i + 1)))
            meta = {"wall_s": p["wall_s"], "engine": engine_str,
                    "model_name": model_name, "ps": ps}
        if isinstance(sampler, GpuSampler):
            sampler.mark_pass(p)
        keep_log(lp("pass{}.meta.json".format(i + 1)),
                 json.dumps(meta, indent=1))
        passes.append(p)

    tele = sampler.stop() if isinstance(sampler, GpuSampler) else sampler
    if isinstance(sampler, GpuSampler):
        keep_log(lp("telemetry.json"), json.dumps(
            {"summary": tele, "marks": sampler.marks,
             "samples": sampler.samples}, indent=1))

    cold_note = None
    l1, l2 = passes[0]["load_ms"], passes[1]["load_ms"]
    if l1 is not None and l2 is not None and l1 < 2 * l2 + 500:
        cold_note = True

    rep = build_rep(passes)
    state, para = diagnose(passes[0], rep, mode, tele)
    why = attribute_why(state, rep, mode, args.extra)

    explain_part = None
    rates = {
        "prefill": rep["prefill_toks"],
        "decode": rep["decode_toks"],
        "wallclock": rep["wallclock_toks"],
    }
    if args.explain is not None:
        v, ep = classify_number(args.explain, rates)
        explain_part = ("{:.1f} tok/s -> {}".format(args.explain, v), ep)

    block = render_verdict(mach, engine_str, model_name, passes, state,
                           para, mode, explain_part, cold_note, why,
                           block_ctx, args.extra, tele)
    print(colorize(block))

    if mode == "server" and url_is_local(binpath) \
            and not rep.get("model_bytes"):
        # a loopback server's weights are a local file: its size is the
        # one calibration figure the http api cannot give plan
        try:
            rep["model_bytes"] = os.path.getsize(
                server_props(binpath).get("model_path") or "")
        except OSError:
            pass
    save_cache({
        "stamp": time.strftime("%Y-%m-%d %H:%M"),
        "model_name": model_name,
        "machine": "{}, {} GB".format(mach["chip"], mach["ram_gb"] or "?"),
        "protocol": PROTOCOL,
        "rates": rates,
        "state": state,
        # what plan's speed estimate calibrates from: decode x bytes is
        # this machine's effective bandwidth, but only on a dense model
        "model_bytes": rep.get("model_bytes"),
        "moe": (bool(rep["n_expert"]) if rep.get("n_expert") is not None
                else None),
        # the runtime kv dtype this run actually used (llama.cpp
        # stderr only); the id card cites it rather than assuming f16
        "kv_types": rep.get("kv_types"),
    })

    if args.json:
        print(json.dumps({"machine": mach, "engine": engine_str,
                          "model": model_name, "mode": mode,
                          "protocol": PROTOCOL, "passes": passes,
                          "warm_median": rates, "state": state,
                          "why": why, "telemetry": tele}, indent=1))

    codes = {"HEALTHY": 0, "NO PLACEMENT EVIDENCE": 0,
             "PARTIAL OFFLOAD": 3, "SILENT CPU FALLBACK": 4,
             "CONFLICTING EVIDENCE": 5}
    sys.exit(codes.get(state, 0))


if __name__ == "__main__":
    main()
