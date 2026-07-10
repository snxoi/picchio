<div align="center">

<img src="assets/picchio-mark-a.svg" width="96" alt="pixel woodpecker on a trunk">

<h1>picchio</h1>

<p>Picchio is Italian for woodpecker: one Python file that knocks on
your local LLM setup and listens for hollow spots. Which tok/s did
you actually get, and did the GPU really do the work?</p>

<p>
<a href="https://github.com/snxoi/picchio/actions/workflows/selftest.yml"><img src="https://github.com/snxoi/picchio/actions/workflows/selftest.yml/badge.svg" alt="selftest"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="license: MIT"></a>
<img src="https://img.shields.io/badge/python-3.9%2B%2C%20stdlib%20only-3776ab" alt="python 3.9+, stdlib only">
</p>

<p><a href="#get-it-running">Install</a> · <a href="#the-three-numbers">What it checks</a> · <a href="examples/">Examples</a></p>

<img src="assets/healthy-verdict.svg" width="600" alt="picchio verdict block in a terminal: GPU ENGAGED 33/33 layers, three lanes reported, verdict HEALTHY">

</div>

That block is the whole product: three lanes that never get merged,
placement evidence from the engine's own logs, a breakdown of the cold
pass, and a verdict. It is 15 lines and 66 columns on purpose, so it
survives being pasted into a comment thread. Real output, unedited
([examples/healthy-metal.txt](examples/healthy-metal.txt)); the text
version below is the one you paste:

```
model    Qwen3.5-9B-Q4_K_M.gguf, 8.95 B, 5.28 GiB, llama.cpp b9430
gpu      ENGAGED: 33/33 layers on GPU (Metal: Apple M5)
                 prefill         decode      wallclock
  cold       597.3 tok/s     17.2 tok/s     11.4 tok/s
  warm mid   558.9 tok/s     20.0 tok/s     14.4 tok/s
  warm span      522~596      18.8~21.2      13.2~15.6
where the cold pass went (11.1 s, 4/10 threads)
  load weights    1.8 s  #####.......................   16%
  prefill         1.3 s  ###.........................   11%
  decode          7.4 s  ###################.........   66%
  engine misc     0.7 s  ##..........................    6%
VERDICT: HEALTHY. The GPU did the work. Quote the warm median
  decode: 20.0 tok/s. 559 tok/s is prefill: reading speed, not
  writing.
-- picchio v0.1.0 mp1 on Apple M5, 32 GB, macOS 26.5.1
```

## Get it running

```
curl -LO https://raw.githubusercontent.com/snxoi/picchio/main/picchio.py
python3 picchio.py /path/to/model.gguf     # llama.cpp, full diagnosis
python3 picchio.py qwen3.5:9b              # ollama tag, measurement mode
```

No pip, no dependencies, no config. One Python file, 904 lines,
stdlib only. If
you have python3 plus either llama.cpp or ollama, you already have
everything it needs. It runs your model three times with a fixed
prompt (the first pass cold, the rest warm), reads the engine's own
numbers, and prints the verdict block above.

Cost of a run: three passes, about a minute on this machine with the
GPU engaged, a few minutes on CPU. Read heavy; it reads your model file,
writes one small cache file under `~/.cache/picchio`, modifies nothing,
and leaves no process running afterwards.

When to rerun it: after a llama.cpp or ollama upgrade, after an OS
update, after switching quants of the same model, after touching -ngl
or context size, and once before you post a tok/s number anywhere.

The parser is pinned by the raw logs in this repo. Running
`python3 picchio.py --selftest` replays the unedited engine output in
[examples/raw/](examples/raw/) and has to reproduce every committed
verdict block line for line; right now that is 9 pass fixtures and 3
blocks, and the badge above runs exactly that on every push.

## Three findings set the tone

The GPU bought 22x on prefill but only 2x on decode, so the number
everyone posts is blind to the bottleneck. A 36 tok/s reading I
trusted reproduced in zero of 32 matrix cells. A background download
cut decode nearly in half, which is why you benchmark on an idle
machine. Everything below unpacks these three.

## The three numbers

Every tok/s figure belongs to one of three lanes, and picchio never
merges the three lanes into one number.

Prefill is how fast the model reads your prompt. Decode is how fast it
writes the answer. Wallclock is generated tokens divided by everything,
load and warmup included, which is what your stopwatch and your gut
measure. On the machine above the warm medians land at 559, 20.0 and
14.4. On the CPU run below they are 26, 11 and 3. A single unlabeled
number spanning a 30x range is not a measurement, it is a rumor.

When a screenshot shows a Mac doing 500 tok/s, that is almost always
prefill. When llama-bench prints tg128, that is decode. When an app
feels slow before the first word appears, that is cold load plus
prefill, and no decode number will explain it. Before you post your next
tok/s number, it costs one minute to run this and find out which lane it
lives in.

## The number everyone posts cannot see your bottleneck

Measured here, the GPU buys about 2x on decode and about 22x on prefill
(both runs are in [examples/](examples/), 4 of 10 cpu threads on the CPU
side). Nearly every tok/s figure posted online is decode, because decode
is the one that feels like typing speed. But on consumer hardware the
pain lives mostly in prefill: it decides how long a long prompt sits
silent before the first word. Two setups can post the same decode number
while one takes ten times longer to start answering. And if your engine
quietly fell back to CPU, decode is the number that will not tell
you.

## The hollow spot: silent CPU fallback

Same machine, same model, same file, forced to CPU
([examples/cpu-fallback.txt](examples/cpu-fallback.txt)):

```
model    Qwen3.5-9B-Q4_K_M.gguf, 8.95 B, 5.28 GiB, llama.cpp b9430
gpu      NOT ENGAGED: 0/33 layers on GPU
                 prefill         decode      wallclock
  cold        24.3 tok/s      9.3 tok/s      2.7 tok/s
  warm mid    25.7 tok/s     11.2 tok/s      2.9 tok/s
  warm span        26~26      11.0~11.4        2.9~2.9
where the cold pass went (47.9 s, 4/10 threads, weights cached)
  load weights    2.0 s  #...........................    4%
  prefill        31.4 s  ##################..........   65%
  decode         13.7 s  ########....................   29%
  engine misc     0.9 s  #...........................    2%
VERDICT: SILENT CPU FALLBACK. 0 of 33 layers reached the GPU.
  Decode (11.2) looks passable, which is how this hides. Prefill
  at 26 tok/s parks a 2500 token prompt 97 s out. Check -ngl.
-- picchio v0.1.0 mp1 on Apple M5, 32 GB, macOS 26.5.1
```

Look at what moved and what did not. Decode dropped 2x, which in a chat
you might shrug at. Prefill dropped 22x, and the first word of a long
prompt now takes minutes. picchio calls this from two directions at
once: the engine's own layer placement log (0/33 offloaded) and the
prefill signature. You can reproduce this verdict on any Apple Silicon
machine with:

```
python3 picchio.py model.gguf -- --device none -ngl 0
```

Anything after the bare `--` goes straight to the llama.cpp binary.

## The number you saw somewhere

The third thing picchio does is interrogate a number for you. Someone
posts a tok/s figure, or you remember one, and you want to know what it
probably was:

```
python3 picchio.py --explain 36
```

```
YOUR NUMBER: 36.0 tok/s -> MATCHES NOTHING MEASURED HERE
  36.0 tok/s is not within 30% of anything measured here
  (closest: decode, off by 1.8x; measured: prefill 558.9, decode
  20.0, wallclock 14.4 tok/s). Before trusting that number, ask
  which of the three rates it was, and on what hardware, quant,
  and context length.
(rates: Qwen3.5-9B-Q4_K_M.gguf, Apple M5, 32 GB, 2026-07-11)
```

That 36 is the exact number this repo exists because of (the story
is in "Why I wrote this" below), asked against the machine it
supposedly came from. This short check is its own output,
deliberately not a verdict block: picchio caches the rates from your
last diagnostic run, so the check needs no rerun. Pass `--explain`
together with a model path instead and the same section is appended
under a full verdict block, one run for both.

## Ollama mode

Give picchio an ollama model tag instead of a file path and it runs the
same passes through your local ollama server (default
`127.0.0.1:11434`, or set `OLLAMA_HOST`). You get the same three lanes,
the same cold pass breakdown, and a placement check based on the memory
split ollama itself reports: how much of the model sits in GPU memory
versus CPU memory.

Real run, same weights imported into ollama
([examples/ollama-qwen35.txt](examples/ollama-qwen35.txt)):

```
model    qwen3.5:9b, 9.0 B, Q4_K_M, 5.55 GiB, ollama 0.31.1
gpu      ENGAGED: 100% of weights in GPU memory (ollama ps)
                 prefill         decode      wallclock
  cold       539.2 tok/s     19.3 tok/s     12.2 tok/s
  warm mid   853.5 tok/s     19.9 tok/s     17.1 tok/s
  warm span      847~860      19.5~20.4      16.8~17.4
where the cold pass went (10.5 s)
  load weights    2.5 s  #######.....................   23%
  prefill         1.4 s  ####........................   14%
  decode          6.6 s  ##################..........   63%
  engine misc     0.0 s  ............................    0%
VERDICT: HEALTHY. Ollama reports 100% of weights in GPU memory.
  Quote the warm median decode: 19.9 tok/s. 853 tok/s is
  prefill: reading, not writing.
-- picchio v0.1.0 mp1 on Apple M5, 32 GB, macOS 26.5.1
```

Be aware of what this mode cannot see, because ollama does not expose
it: per layer placement, device init logs, and thread configuration.
That is why llama.cpp mode is the full diagnosis and ollama mode is
measurement plus a placement check. If ollama gives no memory split at
all, picchio reports the placement as unknown instead of guessing. And
because a reported split can itself be wrong, picchio cross checks it
against the measured rates: when ollama claims full GPU placement but
the prefill to decode ratio looks CPU shaped, the verdict downgrades
to CONFLICTING EVIDENCE instead of HEALTHY.
These two modes are the whole scope; picchio stays one readable file.

## Options

```
picchio MODEL [flags] [-- engine args]

MODEL            a .gguf path (llama.cpp) or an ollama model tag
--passes N       measurement passes, first one cold (default 3)
--explain TOKS   classify a number you saw against the measured lanes
--keep-logs DIR  save each pass's raw engine output into DIR
--json           machine readable measurements after the block
--bin PATH       choose the llama.cpp binary yourself
--selftest       replay examples/raw, verify committed verdicts reproduce
--version        print version and measurement protocol
```

Anything after a bare `--` goes straight to the llama.cpp binary.
Color appears only on a terminal (`NO_COLOR` is respected); piped
output is always plain ASCII, so a pasted block is byte for byte what
the selftest verifies.

## Why I wrote this

I had been systematically measuring local models for an app I am
building, weeks of it, when I nearly filed a bug against my own code.
Bare llama.cpp gave me 36 tok/s. The same model through the app gave
11.5. Same machine, same day, and 3x is the kind of number you
reorganize a week around.

Before writing the fix I reran both sides properly: same binary, same
parameters, a 32 cell matrix across CPU and GPU, cold and warm. The 36
never reproduced. Not in one cell. The slowdown I was about to hunt did
not exist. The number I had trusted was a rate from a different lane,
most likely prefill or a wall clock reading from some other run,
remembered as if it were generation speed. I never wrote down which
lane it came from, so it got to mean whatever my theory needed it to
mean.

What the matrix did surface was a real problem somewhere else entirely.
On some runs the engine put every layer on the CPU without saying
anything at the level you normally look at. Generation speed barely
moved, which is what makes this failure mode invisible. Time to first
token on a long prompt is what explodes: about 5 seconds on the GPU
became about 50 on the CPU for a 2.5k token prompt, on the same
machine.

So there was no 3x slowdown, there was a silent GPU problem, and my
own note was the bug that hid one behind the other. Two measurement
lessons, both invisible in a single tok/s number. picchio is that week
of debugging folded into one file you can run in a minute.

## Is this not just llama-bench?

llama-bench is good and you should use it. It answers a different
question. It tells you how fast this machine can run this model: separate
pp and tg rates, steady state, warmup on by default. picchio tells you
what actually happened on a real run and why it felt the way it felt.

Concretely, measured on this machine, same model, same day:

| tool, config              | prompt side   | generation side | notes                     |
|---------------------------|---------------|-----------------|---------------------------|
| llama-bench, default      | pp256: 597.06 | tg64: 20.21     | backend column: BLAS,MTL  |
| llama-bench, -ngl 0 (CPU) | pp256: 27.82  | tg64: 11.90     | backend column: BLAS,MTL  |

Both rows report the same backend, because that column describes what
the binary was compiled with, not where your tokens were computed. The
21x prompt side collapse is the only visible trace of the CPU run, and
you can only read it if you already know the healthy baseline. There is
also no load time, no cold and warm split, and no interpretation; that
last part is fair, a benchmark is not supposed to have opinions.

picchio exists for the layer under the numbers: was the GPU engaged,
with the engine's own placement evidence attached, where did the first
ten seconds go, and which lane does a given number belong to.

## Measured on this machine

Apple M5, 32 GB, macOS 26.5.1, llama.cpp build 9430 and ollama 0.31.1,
roughly 730 prompt tokens and 128 generated tokens per pass, three
passes, the first one cold. That protocol is named in every block
footer (mp1); if it ever changes, the tag changes, so numbers from
different protocols never sit in one series.

Every number in this table came out of a real run on this hardware;
there are no projected or extrapolated numbers anywhere in this repo.
Ranges span the cold pass and the warm spread of the recorded runs in
[examples/](examples/), and the unedited engine output behind each
example sits in [examples/raw/](examples/raw/), written by the
`--keep-logs` flag: the verdict quotes the numbers, the log is where
they came from.

| config                          | prefill tok/s | decode tok/s | wallclock tok/s |
|---------------------------------|--------------:|-------------:|----------------:|
| Qwen3.5-9B Q4_K_M, Metal 33/33  |     522 - 597 |  17.2 - 21.2 |     11.4 - 15.6 |
| Qwen3.5-9B Q4_K_M, CPU 0/33     |   24.3 - 26.4 |   9.3 - 11.4 |       2.7 - 2.9 |
| qwen3.5:9b via ollama, 100% GPU |     539 - 860 |  19.3 - 20.4 |     12.2 - 17.4 |

Load time for the 5.28 GiB file: 3.3 s the first time it was ever read,
1.7 to 1.8 s after a cache flush, 0.4 s when the weights were still in
the disk cache. picchio prints a note when your pass 1 was not a true cold start,
because a cached load will flatter your first token time. One more
thing measured the hard way while building this: a large download
running in the background cut decode roughly in half on this machine,
so run picchio on a machine that is otherwise idle.

## Verdicts from other machines

I only own one computer, which is why this table is mostly missing.
Run picchio once and paste the verdict block into an issue, even if it
says everything is fine; a boring HEALTHY on hardware I do not have is
still a data point.

If the verdict gets your machine wrong, that is the issue I want most.
Misdiagnosis reports have their own [issue template](.github/ISSUE_TEMPLATE/misdiagnosis-report.md) and go to the top
of the pile, because a diagnostic that misreads machines it has never
met is just a mirror with opinions.

The prefill, decode and wallclock columns hold warm medians, and the
protocol column says which measurement recipe produced them, so rows
stay comparable within a tag.

| machine         | model, engine                      | protocol | prefill | decode | wallclock | verdict |
|-----------------|------------------------------------|----------|--------:|-------:|----------:|---------|
| Apple M5, 32 GB | Qwen3.5-9B Q4_K_M, llama.cpp b9430 | mp1      |   558.9 |   20.0 |      14.4 | HEALTHY |
| Apple M5, 32 GB | qwen3.5:9b, ollama 0.31.1          | mp1      |   853.5 |   19.9 |      17.1 | HEALTHY |
| Apple M5, 32 GB | Qwen3.6-35B-A3B UD-Q4, llama.cpp   | mp1      |   787.3 |   34.4 |      19.1 | HEALTHY |
| Apple M5, 32 GB | qwen3.6:35b-a3b, ollama 0.31.1     | mp1      |  1191.8 |   33.4 |      27.6 | HEALTHY |
| your machine    |                                    |          |         |        |           |         |

The second model taught its own lesson. A 34.7B MoE with about 3B
active parameters decodes 1.7x faster here than the dense 9B (34.4
against 20.0 tok/s), while its 20.6 GiB of weights turn the cold start
into a load problem: 13 of the first pass's 19 seconds went to
loading. Both engines agree within 3% on its decode, and the raw logs
sit behind the rows above.

## Small glossary

Five terms this repo leans on, one line each.

- prefill: the model reading your prompt, in prompt tokens per second. Elsewhere called prompt processing or pp.
- decode: the model writing its answer, one token at a time. Elsewhere called generation, tg, or eval.
- wallclock: generated tokens divided by total elapsed time, load and everything included. The rate a stopwatch sees.
- TTFT: time to first token, how long the screen stays empty. On a cold start this is roughly load plus prefill.
- layer offload: how many model layers were placed on the GPU. 33/33 is a GPU run, 0/33 is a CPU run no matter what the config claimed.

If one of these definitions is wrong, say so in an issue; the lane
discipline stays, the wording is negotiable.

## What it does not do yet

The tested path is one Apple Silicon machine, llama.cpp and ollama.
Linux parsing (CUDA and Vulkan log lines, /proc hardware info) is
written but has not touched real hardware; if you run it there, I want
the verdict block either way.

MLX, LM Studio and remote servers are out of scope. Old llama.cpp
builds are handled with a flag fallback ladder, but very old builds
may only get partial evidence, and picchio will say so rather than
guess.

Passes run back to back, so the first is only a true cold start if the
model was not recently loaded; when the load times give that away, the
block says weights cached. Warm prefill here still carries some spread
(522~596 across two warm passes); more passes tighten the median at
the cost of runtime (`--passes 5`).

Exit codes, for scripting: 0 healthy or no evidence, 2 could not run,
3 partial offload, 4 silent CPU fallback, 5 conflicting evidence.

## License

[MIT](LICENSE).
