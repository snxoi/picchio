#!/usr/bin/env python3
# picchio: knocks on your local LLM setup and listens for hollow spots.
#
# What it does, in one run:
#   1. runs the same fixed prompt through your model N times (default 3;
#      the first pass is the cold one, the rest are warm)
#   2. reads the engine's own timing and placement evidence
#   3. reports prefill, decode and wallclock tok/s as three separate lanes,
#      cold pass first, then the warm median and the warm spread
#   4. tells you whether the GPU actually did the work, or quietly did not
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
#
# Needs: python3 (any recent one), plus llama.cpp on PATH or a local ollama.
# Nothing else. No pip.
#
# Exit codes: 0 ok/healthy, 2 could not run, 3 partial offload,
#             4 silent cpu fallback, 5 conflicting evidence.

import argparse
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
        "model_params": None, "model_size": None,
        "threads": None, "cores": None,
        "vram_frac": None,
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
        if "system_info" in line:
            m = re_threads.search(line)
            if m:
                # llama.cpp defaults to 4 threads on this 10 core test
                # machine; recorded rather than tuned, because CPU rates
                # move a lot with -t and the block should say so.
                d["threads"] = int(m.group(1))
                d["cores"] = int(m.group(2))
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


# --------------------------------------------------------------- diagnosis

def diagnose(cold, rep, mode):
    """Returns (state, paragraph). State drives the exit code. The
    paragraph budget is about 160 characters: the whole block must stay
    inside 15 lines."""
    decode = rep["decode_toks"] or cold["decode_toks"]
    prefill = rep["prefill_toks"] or cold["prefill_toks"]
    wait_s = 2500.0 / prefill if prefill else None

    if mode == "ollama":
        frac = rep["vram_frac"]
        if frac is None:
            return "NO PLACEMENT EVIDENCE", (
                "Ollama did not report a memory split for this model, so "
                "picchio cannot say where it ran. Rates are measured; "
                "placement is not."
            )
        if frac < 0.05:
            para = "Ollama reports 0% of weights in GPU memory."
            if decode:
                para += (" Decode ({:.1f} tok/s) may look passable, which "
                         "is how this hides.".format(decode))
            if prefill:
                para += (" Prefill at {:.0f} tok/s parks a 2500 token "
                         "prompt {:.0f} s out.".format(prefill, wait_s))
            return "SILENT CPU FALLBACK", para
        if frac < 0.95:
            return "PARTIAL OFFLOAD", (
                "Ollama reports {:.0f}% of weights in GPU memory, the rest "
                "on CPU, usually a memory fit call. Expect rates below a "
                "fully offloaded model.".format(frac * 100)
            )
        # ollama's reported split has been known to disagree with where
        # the kernels actually ran, so a full-GPU claim is cross checked
        # against the speed signature before it earns HEALTHY.
        if prefill and decode and prefill < 5 * decode:
            return "CONFLICTING EVIDENCE", (
                "Ollama says 100% GPU, but prefill at {:.0f} tok/s is "
                "only {:.1f}x decode, a CPU shaped ratio. Believe neither "
                "signal: check the ollama server log.".format(
                    prefill, prefill / decode)
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
        para = "0 of {} layers reached the GPU.".format(total)
        if decode:
            para += (" Decode ({:.1f}) looks passable, which is how this "
                     "hides.".format(decode))
        if prefill:
            para += (" Prefill at {:.0f} tok/s parks a 2500 token prompt "
                     "{:.0f} s out. Check -ngl.".format(prefill, wait_s))
        return "SILENT CPU FALLBACK", para
    if total and n < total:
        return "PARTIAL OFFLOAD", (
            "{} of {} layers fit on the GPU, the rest run on CPU, usually "
            "a memory fit call. Expect rates below a fully offloaded "
            "model.".format(n, total)
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
        elif line.startswith("-- picchio") or (
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
                   explain_part=None, cold_note=None):
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
    out.append("gpu      " + gpu_line(rep, mode))
    out.append("           {:>13}  {:>13}  {:>13}".format(
        "prefill", "decode", "wallclock"))
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
    out.extend(textwrap.wrap("VERDICT: {}. {}".format(state, para),
                             width=WIDTH - 2, subsequent_indent="  "))
    if explain_part:
        out.append("YOUR NUMBER: " + explain_part[0])
        out.extend(wrap_para(explain_part[1]))
    out.append("-- picchio v{} {} on {}, {} GB, {}".format(
        VERSION, PROTOCOL, mach["chip"], mach["ram_gb"] or "?", mach["os"]))
    return "\n".join(out)


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
        state, para = diagnose(passes[0], build_rep(passes), mode)
        got = render_verdict(
            machine_info(), metas[0].get("engine", "?"),
            metas[0].get("model_name", "?"), passes, state, para, mode,
            None, cold_note).splitlines()
        if got[:-1] == want[:-1]:
            rp_ok += 1
        else:
            for a, b in zip(want, got):
                if a != b:
                    print("  {} mismatch:\n    want: {}\n    got:  {}".format(
                        name, a, b))
                    break
    print("parser fixtures {}/{}, verdict replay {}/{}".format(
        fx_ok, fx_all, rp_ok, rp_all))
    sys.exit(0 if fx_ok == fx_all and rp_ok == rp_all and rp_all else 1)


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
    if os.path.isfile(args.model):
        mode = "llama.cpp"
        binpath = find_binary(args.bin)
        engine_str = "llama.cpp " + engine_version(binpath)
        model_name = os.path.basename(args.model)
        for i in range(args.passes):
            sys.stderr.write("picchio: pass {}{} ...\n".format(
                i + 1, " (includes any cold load)" if i == 0 else " (warm)"))
            p = run_llama_pass(binpath, args.model, args.extra,
                               lp("pass{}.stderr.txt".format(i + 1)))
            keep_log(lp("pass{}.meta.json".format(i + 1)), json.dumps(
                {"wall_s": p["wall_s"], "engine": engine_str,
                 "model_name": model_name}, indent=1))
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
        for i in range(args.passes):
            sys.stderr.write("picchio: pass {}{} ...\n".format(
                i + 1, " (includes any cold load)" if i == 0 else " (warm)"))
            p, ps = run_ollama_pass(
                args.model, lp("pass{}.response.json".format(i + 1)))
            keep_log(lp("pass{}.meta.json".format(i + 1)), json.dumps(
                {"wall_s": p["wall_s"], "engine": engine_str,
                 "model_name": model_name, "ps": ps}, indent=1))
            passes.append(p)

    cold_note = None
    l1, l2 = passes[0]["load_ms"], passes[1]["load_ms"]
    if l1 is not None and l2 is not None and l1 < 2 * l2 + 500:
        cold_note = True

    rep = build_rep(passes)
    state, para = diagnose(passes[0], rep, mode)

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
                           para, mode, explain_part, cold_note)
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
                          "warm_median": rates, "state": state},
                         indent=1))

    codes = {"HEALTHY": 0, "NO PLACEMENT EVIDENCE": 0,
             "PARTIAL OFFLOAD": 3, "SILENT CPU FALLBACK": 4,
             "CONFLICTING EVIDENCE": 5}
    sys.exit(codes.get(state, 0))


if __name__ == "__main__":
    main()
