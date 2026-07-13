<div align="center">

<img src="assets/picchio-mark-a.svg" width="96" alt="pixel woodpecker on a trunk">

<h1>picchio</h1>

<p>Picchio is Italian for woodpecker: one Python file that answers
two questions about your local LLM setup: which tok/s did you
actually get, and did the GPU really do the work?</p>

<p>
<a href="https://github.com/logxio/picchio/actions/workflows/selftest.yml"><img src="https://github.com/logxio/picchio/actions/workflows/selftest.yml/badge.svg" alt="selftest"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="license: MIT"></a>
<img src="https://img.shields.io/badge/python-3.9%2B%2C%20stdlib%20only-3776ab" alt="python 3.9+, stdlib only">
</p>

<p><a href="#get-it-running">Install</a> · <a href="#commands">Commands</a> · <a href="#the-three-numbers">What it checks</a> · <a href="#measured-on-this-machine">Measured</a> · <a href="examples/">Examples</a></p>

<img src="assets/picchio-demo.svg" width="600" alt="animated terminal replay: python3 picchio.py finds two models, runs three passes, and prints the 15 line verdict block, verdict HEALTHY">

<p>A real run, replayed. Time compressed.</p>

</div>

That block is the product: 15 lines, 66 columns, narrow enough to
paste into a comment thread. Real output, unedited
([examples/healthy-metal.txt](examples/healthy-metal.txt));
the text version below is the one you paste:

```
model    Qwen3.5-9B-Q4_K_M.gguf, 8.95 B, 5.28 GiB, llama.cpp b9430
gpu      ENGAGED: 33/33 layers on GPU (Metal: Apple M5)
os       gpu idle 0%, work 99%, mem +6.0 GiB, 11.0 W
ctx 4096         prefill         decode      wallclock
  cold       584.9 tok/s     21.0 tok/s     13.1 tok/s
  warm mid   588.0 tok/s     21.1 tok/s     15.5 tok/s
  warm span      585~591      21.0~21.2      15.4~15.5
where the cold pass went (9.7 s, 4/10 threads)
  load weights    1.8 s  #####.......................   18%
  prefill         1.3 s  ####........................   13%
  decode          6.1 s  #################...........   62%
  engine misc     0.6 s  ##..........................    6%
VERDICT: HEALTHY. The GPU did the work. Quote the warm median
  decode: 21.1 tok/s.
-- picchio v0.1.0 mp1 on Apple M5, 32 GB, macOS 26.5.1
```

## Get it running

```
curl -LO https://raw.githubusercontent.com/logxio/picchio/main/picchio.py
python3 picchio.py
```

That second line, with no arguments, looks around your machine
(ollama tags, the current folder, the HF and LM Studio caches). One
model found, it runs it; several, it lists them and you pick by
number, path, or tag; none, it asks for a path. It asks once, then
runs to the verdict block without stopping. Piped or redirected, it
prints the commands instead. Or point it at a model yourself:
a .gguf path (`python3 picchio.py /path/to/model.gguf`) gets the
full llama.cpp diagnosis, an ollama tag (`python3 picchio.py
qwen3.5:9b`) gets measurement mode.

No pip, no dependencies, no config. One Python file, 3432 lines,
stdlib only; python3 plus either llama.cpp or ollama is everything
it needs. It runs your model three times with a fixed prompt (the
first pass cold, the rest warm), reads the engine's own numbers
while a background thread reads the OS's GPU meter, and prints the
block above. A run costs about a minute here with the GPU engaged,
a few minutes on CPU; it writes one small cache file under
`~/.cache/picchio`, modifies nothing, and leaves no process behind.

When to rerun it: after a llama.cpp or ollama upgrade, after an OS
update, after switching quants of the same model, after touching -ngl
or context size, and once before you post a tok/s number anywhere.

`python3 picchio.py --selftest` replays the raw engine logs in
[examples/raw/](examples/raw/) and must reproduce every committed
verdict block line for line; the badge runs it on every push.

## Commands

| command | what it does | real output |
|---------|--------------|-------------|
| `picchio model.gguf` | full llama.cpp diagnosis: three passes, placement, cold start breakdown, verdict | [example](examples/healthy-metal.txt) |
| `picchio qwen3.5:9b` | same passes through your local ollama server, placement from the memory split it reports | [example](examples/ollama-qwen35.txt) |
| `picchio http://127.0.0.1:8080` | measures a llama-server already running, nothing launched, warm rows only | [example](examples/server-endpoint.txt) |
| `picchio guard -- <command>` | wraps your own command, warns the moment layers land off the GPU, never kills it | [example](examples/guard-ngl0.txt) |
| `picchio compare A.txt B.txt` | diffs two saved blocks variable by variable, the first config difference takes the blame | [example](examples/compare.txt) |
| `picchio verify FILE` | flags a pasted block whose own numbers contradict each other | [example](examples/verify-forged.txt) |
| `picchio watch [PID]` | points the OS GPU meter at a process or the whole GPU, no engine log parsing (macOS) | [example](examples/watch-ollama.txt) |
| `picchio plan [MODEL]` | will it fit, priced from the gguf header; a decode estimate appears once one run is measured | [example](examples/plan-35b.txt) |
| `picchio --explain 36` | classifies a number you saw against the lanes measured here (cached rates, no rerun) | [example](examples/explain-36.txt) |
| `picchio model.gguf --ctx-sweep` | re-measures the lanes at several context depths and reports the decay slope | [example](examples/ctx-sweep.txt) |

```
--passes N       measurement passes, first one cold (default 3)
--keep-logs DIR  save each pass's raw engine output into DIR, plus
                 the sampled GPU curve (telemetry.json) on macOS
--no-telemetry   skip the OS-side GPU sampling; the os line then
                 says the verdict rests on engine+timing only
--json           machine readable measurements after the block
--bin PATH       choose the llama.cpp binary yourself
--selftest       replay examples/raw, verify committed verdicts reproduce
--version        print version and measurement protocol
```

Anything after a bare `--` goes straight to the llama.cpp binary.
Color only on a terminal (`NO_COLOR` respected); piped output is
plain ASCII.

Exit codes, for scripting: 0 healthy or no evidence, 2 could not
run, 3 partial offload, 4 silent CPU fallback, 5 conflicting
evidence. guard passes the wrapped command's own exit code through
(128 plus the signal number if it died by one); compare exits 0
once both blocks parse; verify exits 0 when a block is
self-consistent, 5 when its sources fight; watch exits 0 when the
GPU is working, 4 when it sits idle.

## The three numbers

Every tok/s figure belongs to one of three lanes, and picchio never
merges them. Prefill (elsewhere called prompt processing or pp) is
how fast the model reads your prompt; decode (tg or eval) is how
fast it writes the answer; wallclock is generated tokens divided by
everything, load and warmup included, which is what your stopwatch
measures. In the block above the warm medians land at 588, 21.1 and
15.5; on the CPU run below they land at 27, 12 and 3. A tok/s
number without its lane label cannot be compared with anything.

<p align="center">
<img src="assets/prefill-decode-asymmetry.svg" width="600" alt="prefill collapses 22x from GPU to CPU while decode only drops 1.7x on the same model and file">
</p>

The lanes fail separately. Measured here, the GPU buys about 22x on
prefill and under 2x on decode (both runs are in
[examples/](examples/), 4 of 10 cpu threads on the CPU side).
Nearly every figure posted online is decode, but prefill sets the
time to first token on a long prompt: a Mac screenshot showing 500
tok/s is almost always prefill.

## Silent CPU fallback

Same machine, same model, same file, forced to CPU
([examples/cpu-fallback.txt](examples/cpu-fallback.txt)):

<p align="center">
<img src="assets/cpu-fallback-verdict.svg" width="600" alt="picchio verdict block in a terminal: NOT ENGAGED 0/33 layers, OS meter flat, verdict SILENT CPU FALLBACK, WHY line naming the forcing flags">
</p>

The text version:

```
model    Qwen3.5-9B-Q4_K_M.gguf, 8.95 B, 5.28 GiB, llama.cpp b9430
gpu      NOT ENGAGED: 0/33 layers on GPU [--device none -ngl 0]
os       gpu idle 8%, work 5%, mem +0.3 GiB, 0.1 W
ctx 4096         prefill         decode      wallclock
  cold        22.8 tok/s      9.3 tok/s      2.5 tok/s
  warm mid    26.8 tok/s     12.2 tok/s      3.0 tok/s
  warm span        27~27      12.0~12.4        3.0~3.0
where the cold pass went (49.9 s, 4/10 threads, weights cached)
  load weights    2.1 s  #...........................    4%
  prefill        33.4 s  ###################.........   67%
  decode         13.7 s  ########....................   27%
  engine misc     0.8 s  ............................    2%
VERDICT: SILENT CPU FALLBACK. Prefill: 93 s per 2500 tokens.
WHY: forced by flag: --device none -ngl 0
-- picchio v0.1.0 mp1 on Apple M5, 32 GB, macOS 26.5.1
```

Decode barely dropped, but prefill fell 22x: the first word of a
long prompt now takes a minute and a half. The WHY line names the
first cause the run's own evidence can prove, or says unknown.

While measuring local models for an app I am building, weeks of
it, bare llama.cpp gave me 36 tok/s and the same model through the
app gave 11.5: that gap is why this repo exists. A 32 cell matrix
across CPU and GPU, cold and warm, reproduced the 36 in no cell, a
rate from a different lane remembered as generation speed. What
the matrix did surface was this silent fallback.

## The os line

Engine logs have been wrong before: ollama has shipped releases
that reported a full GPU load while the kernels ran elsewhere. So
while the passes run on macOS, a background thread reads the OS's
own GPU accounting (`ioreg`, 4 Hz) and GPU power from the same
energy counters `powermetrics` reports, minus the sudo: the `os`
line. HEALTHY requires the engine's log, the OS meter and the
speed signature to agree; a full offload claim over a GPU the OS
saw stay flat gets CONFLICTING EVIDENCE (exit 5). A missing source
abstains, and the line says which evidence is left.

## Not just llama-bench

llama-bench is good and you should use it; it answers a different
question, how fast this machine can run this model as steady state
pp and tg rates. picchio answers what actually happened on a real
run. Measured on this machine, same model, same day:

| tool, config              | prompt side   | generation side | notes                     |
|---------------------------|---------------|-----------------|---------------------------|
| llama-bench, default      | pp256: 597.06 | tg64: 20.21     | backend column: BLAS,MTL  |
| llama-bench, -ngl 0 (CPU) | pp256: 27.82  | tg64: 11.90     | backend column: BLAS,MTL  |

Both rows report the same backend, because that column describes
what the binary was compiled with, not where your tokens were
computed. The 21x prompt side collapse is the CPU run's only
visible trace, readable only if you already know the healthy
baseline; there is no load time, no cold/warm split, no verdict.
picchio reports that layer: placement, cold start, and a verdict.

## Measured on this machine

Apple M5, 32 GB, macOS 26.5.1, llama.cpp build 9430 and ollama
0.31.1, roughly 730 prompt tokens and 128 generated tokens per pass,
three passes, the first one cold. That protocol is named in every
block footer (mp1); if it ever changes the tag changes. Every
number came out of a real run on the machine in its row, the lane
columns hold warm medians, and the raw engine output behind the
first three rows sits in [examples/raw/](examples/raw/), written by
`--keep-logs`.

| machine         | model, engine                      | protocol | prefill | decode | wallclock | verdict             |
|-----------------|------------------------------------|----------|--------:|-------:|----------:|---------------------|
| Apple M5, 32 GB | Qwen3.5-9B Q4_K_M, llama.cpp b9430 | mp1      |   588.0 |   21.1 |      15.5 | HEALTHY             |
| Apple M5, 32 GB | same, forced CPU (0/33 layers)     | mp1      |    26.8 |   12.2 |       3.0 | SILENT CPU FALLBACK |
| Apple M5, 32 GB | qwen3.5:9b, ollama 0.31.1          | mp1      |   833.8 |   21.3 |      18.1 | HEALTHY             |
| Apple M5, 32 GB | Qwen3.6-35B-A3B UD-Q4, llama.cpp   | mp1      |   787.3 |   34.4 |      19.1 | HEALTHY             |
| Apple M5, 32 GB | qwen3.6:35b-a3b, ollama 0.31.1     | mp1      |  1191.8 |   33.4 |      27.6 | HEALTHY             |
| RTX 4090, Linux | Qwen3.5-9B Q4_K_M, llama.cpp b9430 | mp1      |  6763.3 |  138.0 |      25.2 | HEALTHY             |
| your machine    |                                    |          |         |        |           |                     |

I only own one computer (the 4090 was rented for an afternoon to
exercise the Linux path), which is why this table is mostly missing.
Run picchio once and paste the verdict block into an issue, even if
it says everything is fine; a boring HEALTHY on hardware I do not
have is still a data point. A wrong verdict is the issue I want
most: [misdiagnosis reports](.github/ISSUE_TEMPLATE/misdiagnosis-report.md)
go to the top of the pile.

The 35B rows: a 34.7B MoE with about 3B active parameters decodes
1.6x faster here than the dense 9B (34.4 against 21.1 tok/s), while
its 20.6 GiB of weights turn the cold start into a load problem, 13
of that first pass's 19 seconds going to loading. A large download
running in the background cut decode roughly in half, so run
picchio on a machine that is otherwise idle.

## Limits

- The tested path is one Apple Silicon machine (llama.cpp and
  ollama) plus one rented Linux RTX 4090, where the CUDA parsing
  and the verdict held. ollama on Linux and Vulkan log lines have
  not touched real hardware; if you run those, I want the verdict
  block either way. The os line does not sample on Linux yet, so a
  Linux verdict rests on the engine log and the speed signature.
- The full verdict block, with its three lanes and cold-start
  breakdown, is llama.cpp and ollama only. MLX, LM Studio and other
  engines get placement truth through `watch`, not the lane table.
- Ollama mode cannot see per layer placement, device init logs, or
  thread configuration; ollama does not expose them. Placement
  comes from the memory split ollama reports, and reads unknown
  when no split is reported.
- Server mode gets no placement claim from the llama-server api,
  so the judgment rests on the os meter and the speed signature;
  there is no cold row (the server already owns the weights), each
  pass forces a full prompt read (per request cache off), and on a
  remote url the os line says not sampled, the footer names the
  machine picchio ran on, and wallclock includes the network round
  trip.
- Very old llama.cpp builds may only get partial evidence; the
  block names whatever is missing.
- Passes run back to back, so the first is only a true cold start
  if the model was not recently loaded; the block then says weights
  cached, because a cached load flatters your first token time.
- Warm numbers drift between sessions: the 9B medians in this repo
  moved 5 to 8% between two recording rounds on an idle machine.
  More passes (`--passes 5`) tighten a single reading.
- The os meter counts the whole GPU, not one process, so it only
  judges runs that started from an idle GPU.
- The watts come from a private macOS framework (the same counters
  powermetrics prints); an OS update can move it, in which case the
  watts drop off the line and everything else keeps working.

## License

[MIT](LICENSE).
