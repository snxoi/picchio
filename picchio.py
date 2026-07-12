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
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
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
        "vram_frac": None,
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


def engine_version(binpath):
    out = _cmd_out([binpath, "--version"])
    m = re.search(r"version:\s*(\S+)\s*\(([0-9a-f]+)\)", out)
    if m:
        return "b" + m.group(1)
    return os.path.basename(binpath)


def run_llama_pass(binpath, model, extra_args, log_path=None):
    base = [
        binpath,
        "-m", model,
        "-p", BENCH_PROMPT,
        "-n", str(N_PREDICT),
        "-c", str(CTX),
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
    re_params = re.compile(r"model params\s*=\s*([\d.]+\s*\S?)")
    re_size = re.compile(r"file size\s*=\s*([\d.]+\s*\S+)")
    re_threads = re.compile(r"n_threads\s*=\s*(\d+).*?/\s*(\d+)")
    # e.g. "using device MTL0 (Apple M5) (unknown id) - 25558 MiB free":
    # the free figure the engine itself saw, kept for WHY attribution.
    re_free = re.compile(r"-\s*(\d+)\s*MiB free")

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


def run_ollama_pass(tag, log_path=None):
    t0 = time.monotonic()
    resp = ollama_api("/api/generate", {
        "model": tag,
        "prompt": BENCH_PROMPT,
        "stream": False,
        "options": {"num_predict": N_PREDICT, "num_ctx": CTX, "seed": 7},
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


def discover_models():
    """No model argument: look around this machine (read only, fast) and
    print commands that can be copied as they are."""
    rows = []
    ver = ollama_reachable()
    if ver:
        try:
            for m in ollama_api("/api/tags", timeout=5).get("models", []):
                if m.get("name"):
                    rows.append((m["name"], "ollama"))
        except (urllib.error.URLError, OSError, ValueError):
            pass
    else:
        # ollama not running: its manifest folder still names the tags
        base = os.path.expanduser("~/.ollama/models/manifests")
        for reg in glob.glob(os.path.join(base, "*", "*", "*", "*")):
            parts = reg.split(os.sep)
            name, tag = parts[-2], parts[-1]
            rows.append(("{}:{}".format(name, tag),
                         "ollama, not running"))
    rows = rows[:8]

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
            ggufs.append(f)
    ggufs = ggufs[:8]

    if not rows and not ggufs:
        print("picchio: no model given, and none found in the usual "
              "places\n(no ollama tags, no .gguf in the current folder, "
              "the HF cache,\nor the LM Studio folders).\n\n"
              "Point it at any .gguf file or ollama tag:\n"
              "  python3 picchio.py /path/to/model.gguf\n"
              "  python3 picchio.py some-tag:latest")
        sys.exit(0)
    print("picchio: no model given. Runnable on this machine:\n")
    for tag, note in rows:
        print("  python3 picchio.py {:<36} ({})".format(tag, note))
    for f in ggufs:
        q = '"{}"'.format(f) if " " in f else f
        print("  python3 picchio.py {}".format(q))
    print("\nPick one, or point it at any other .gguf path or ollama "
          "tag.")
    sys.exit(0)


def ollama_unload(tag):
    # Unload first so the cold pass pays the true load cost; ollama
    # keeps models resident for 5 minutes by default, and a cold number
    # measured against a resident model means nothing.
    try:
        ollama_api("/api/generate", {"model": tag, "keep_alive": 0},
                   timeout=60)
    except (urllib.error.URLError, OSError, ValueError):
        pass


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
    instead of quietly reading like a fully instrumented one."""
    if disabled:
        return {"off": "disabled"}
    if platform.system() != "Darwin":
        return {"off": "not macos"}
    if not shutil.which("ioreg"):
        return {"off": "no ioreg"}
    first = read_gpu_stats()
    if first is None:
        return {"off": "ioreg gave no gpu stats"}
    return GpuSampler(first)


class GpuSampler:
    def __init__(self, first):
        self.samples = [first]
        self.marks = []
        try:
            self._power = _IOReport()
        except Exception:
            self._power = None  # private API absent or moved: no watts
        self._hot = thermal_raised()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        period = 1.0 / TELE_HZ
        while not self._stop.is_set():
            tick = time.monotonic()
            s = read_gpu_stats()
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
        return telemetry_summary(self.samples, self.marks,
                                 self._hot or thermal_raised())


def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def telemetry_summary(samples, marks, hot=False):
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
        "hz": TELE_HZ, "n": len(samples),
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


def os_line(tele):
    """The one line of OS evidence in the block, None only when the
    render has no telemetry context at all (pre-telemetry replays)."""
    if tele is None:
        return None
    if tele.get("off"):
        return "gpu not sampled ({}); evidence: engine+timing".format(
            tele["off"])
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
    else:
        n, total = rep["offload_n"], rep["offload_total"]
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


def colorize(text):
    """ANSI color for terminals only. Piped or redirected output stays
    pure ASCII, so a pasted block is identical to what the parser and
    the selftest see. NO_COLOR is respected."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"
    states = (("SILENT CPU FALLBACK", RED), ("CONFLICTING EVIDENCE", YELLOW),
              ("PARTIAL OFFLOAD", YELLOW), ("NO PLACEMENT EVIDENCE", YELLOW),
              ("HEALTHY", GREEN))
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
        elif line.startswith(("WHY: ", "-- picchio")) or (
                "prefill" in line and "wallclock" in line
                and "tok/s" not in line):
            line = DIM + line + RESET
        elif line.startswith("YOUR NUMBER: "):
            line = BOLD + line + RESET
        out.append(line)
    return "\n".join(out)


def gpu_line(rep, mode):
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
    out.append("  {:<9}{:>13}  {:>13}  {:>13}".format(
        "cold", fmt_rate(cold["prefill_toks"]), fmt_rate(cold["decode_toks"]),
        fmt_rate(cold["wallclock_toks"])))
    pm, plo, phi = warm_stats(passes, "prefill_toks")
    dm, dlo, dhi = warm_stats(passes, "decode_toks")
    wm, wlo, whi = warm_stats(passes, "wallclock_toks")
    out.append("  {:<9}{:>13}  {:>13}  {:>13}".format(
        "warm mid", fmt_rate(pm), fmt_rate(dm), fmt_rate(wm)))
    out.append("  {:<9}{:>13}  {:>13}  {:>13}".format(
        "warm span", fmt_span(plo, phi, big=True), fmt_span(dlo, dhi),
        fmt_span(wlo, whi)))

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
                sys.stderr.write(
                    guard_state_line(rep, guard_why(rep, cmd)) + "\n")
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
    sys.stderr.write("\n".join(out) + "\n")
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
    print(render_compare(argv, blocks[0], blocks[1]))


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
    print(render_verify(src, b, verdict, flags))
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
    print(render_watch(ctx, summ, state, para))
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
            None, cold_note, why, effective_ctx(extra), extra,
            tele).splitlines()
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

    def synth_tele(idle_dev, work_dev, mem_base, mem_peak):
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
            "prompt_s": prompt_s, "eval_s": eval_s}])

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
    print("parser fixtures {}/{}, verdict replay {}/{}, compare {}/{}, "
          "telemetry {}/{}, verify {}/{}, watch {}/{}".format(
              fx_ok, fx_all, rp_ok, rp_all, cp_ok, cp_all,
              te_ok, te_all, ve_ok, ve_all, wa_ok, wa_all))
    sys.exit(0 if fx_ok == fx_all and rp_ok == rp_all and rp_all
             and cp_ok == cp_all and te_ok == te_all
             and ve_ok == ve_all and wa_ok == wa_all else 1)


# -------------------------------------------------------------------- main

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
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", nargs="?",
                    help="path to a .gguf file, or an ollama model tag")
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
    ap.add_argument("--selftest", action="store_true",
                    help="replay examples/raw through the parser and "
                         "diagnosis; verify the committed verdicts reproduce")
    ap.add_argument("extra", nargs="*", default=[],
                    help="args after -- go straight to the llama.cpp engine "
                         "(e.g. -- --device none -ngl 0)")
    args = ap.parse_args()

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
        discover_models()
    if args.passes < 2:
        sys.exit("picchio: --passes must be at least 2 (one cold, one warm).")
    if args.extra and "--" not in sys.argv[1:]:
        sys.exit("picchio: unexpected extra arguments: {}\n"
                 "(a pasted trailing comment does this; engine args need "
                 "a bare -- first)".format(" ".join(args.extra)))

    mach = machine_info()
    logdir = args.keep_logs
    lp = (lambda name: os.path.join(logdir, name)) if logdir else \
        (lambda name: None)

    passes = []
    sampler = None
    if os.path.isfile(args.model):
        mode = "llama.cpp"
        binpath = find_binary(args.bin)
        engine_str = "llama.cpp " + engine_version(binpath)
        model_name = os.path.basename(args.model)
        sampler = telemetry_start(args.no_telemetry)
        if isinstance(sampler, GpuSampler):
            time.sleep(1.2)  # a few ticks of idle baseline before pass 1
        for i in range(args.passes):
            sys.stderr.write("picchio: pass {}{} ...\n".format(
                i + 1, " (includes any cold load)" if i == 0 else " (warm)"))
            p = run_llama_pass(binpath, args.model, args.extra,
                               lp("pass{}.stderr.txt".format(i + 1)))
            if isinstance(sampler, GpuSampler):
                sampler.mark_pass(p)
            keep_log(lp("pass{}.meta.json".format(i + 1)), json.dumps(
                {"wall_s": p["wall_s"], "engine": engine_str,
                 "model_name": model_name, "extra_args": args.extra},
                indent=1))
            passes.append(p)
    elif not looks_like_tag(args.model):
        sys.exit("picchio: no such file: {}\nRun picchio with no "
                 "arguments to see the models on this machine.".format(
                     args.model))
    else:
        ver = ollama_reachable()
        if not ver:
            sys.exit(
                "picchio: {!r} looks like an ollama tag, but no ollama "
                "answered at {}.\nStart ollama, or give a .gguf path; "
                "run picchio with no arguments to see both.".format(
                    args.model, OLLAMA_HOST))
        if not ollama_has_model(args.model):
            sys.exit("picchio: ollama at {} does not know the model "
                     "{!r}.\nRun picchio with no arguments to list "
                     "what it does know.".format(OLLAMA_HOST, args.model))
        if args.extra:
            sys.exit("picchio: passthrough args after -- only work in "
                     "llama.cpp mode.")
        mode = "ollama"
        engine_str = "ollama " + ver
        model_name = args.model
        if ollama_ps_entry(args.model):
            sys.stderr.write("picchio: unloading model for a colder "
                             "pass 1 ...\n")
            ollama_unload(args.model)
        sampler = telemetry_start(args.no_telemetry)
        if isinstance(sampler, GpuSampler):
            time.sleep(1.2)  # a few ticks of idle baseline before pass 1
        for i in range(args.passes):
            sys.stderr.write("picchio: pass {}{} ...\n".format(
                i + 1, " (includes any cold load)" if i == 0 else " (warm)"))
            p, ps = run_ollama_pass(
                args.model, lp("pass{}.response.json".format(i + 1)))
            if isinstance(sampler, GpuSampler):
                sampler.mark_pass(p)
            keep_log(lp("pass{}.meta.json".format(i + 1)), json.dumps(
                {"wall_s": p["wall_s"], "engine": engine_str,
                 "model_name": model_name, "ps": ps}, indent=1))
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
                           effective_ctx(args.extra), args.extra, tele)
    print(colorize(block))

    save_cache({
        "stamp": time.strftime("%Y-%m-%d %H:%M"),
        "model_name": model_name,
        "machine": "{}, {} GB".format(mach["chip"], mach["ram_gb"] or "?"),
        "protocol": PROTOCOL,
        "rates": rates,
        "state": state,
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
