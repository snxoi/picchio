# picchio

Picchio is Italian for woodpecker. Woodpeckers find hollow wood by knocking
on it and listening. This is a single Python file that knocks on your local
LLM setup and listens for the two most common hollow spots: tok/s numbers
that do not mean what you think they mean, and a GPU that quietly did
nothing while the CPU did all the work.

```
python3 picchio.py /path/to/model.gguf     # llama.cpp, full diagnosis
python3 picchio.py qwen3.5:9b              # ollama tag, measurement mode
```

No pip, no dependencies, no config. If you have python3 plus either
llama.cpp or ollama, you already have everything it needs. It runs your
model twice with a fixed prompt, reads the engine's own numbers, and
prints a verdict block sized to fit in a forum comment.

## Why I wrote this

Last week I had proof that my app was slowing local models down by a
factor of three. Bare llama.cpp gave me 36 tok/s. The same model inside
the app gave 11.5. Same machine, same day, case closed.

Then I reran both sides properly: same binary, same parameters, a 32 cell
matrix across CPU and GPU, cold and warm. The 36 never reproduced. Not in
one cell. The number I had built a theory on was a rate from a different
lane, most likely prefill or a wall clock reading from some other run,
remembered as if it were generation speed. I never wrote down which lane
it came from, so it got to mean whatever my theory needed it to mean.

The real slowdown was somewhere else entirely. On some runs the engine
put every layer on the CPU without saying anything at the level you
normally look at. Generation speed barely moved, which is what makes this
failure mode invisible. Time to first token on a long prompt is what
explodes: about 5 seconds on the GPU became about 50 on the CPU for a
2.5k token prompt, measured on the same machine during that
investigation.

So the app was not 3x slower. My benchmark was lying, and separately, the
GPU was sometimes not working at all. Two different bugs, both mine, both
invisible in a single tok/s number. picchio is that week of debugging
folded into one file you can run in a minute.

## What it prints

Real output from the machine this was built on, unedited
([examples/healthy-metal.txt](examples/healthy-metal.txt)). The block is
15 lines and 66 columns on purpose: it survives being pasted into a
comment thread as is.

```
model    Qwen3.5-9B-Q4_K_M.gguf, 8.95 B, 5.28 GiB, llama.cpp b9430
gpu      ENGAGED: 33/33 layers on GPU (Metal: Apple M5)
                 prefill         decode      wallclock
  pass 1     584.2 tok/s     18.5 tok/s     11.9 tok/s
  pass 2     591.5 tok/s     20.3 tok/s     14.7 tok/s
where pass 1 went (10.6 s wall, 4/10 threads)
  load weights    1.7 s  #####.......................   16%
  prefill         1.3 s  ###.........................   12%
  decode          6.9 s  ##################..........   65%
  engine misc     0.8 s  ##..........................    7%
VERDICT: HEALTHY
  The GPU did the work. Quote decode (20.3 tok/s) when you
  compare setups. 591 tok/s is real too, but it is prefill:
  reading speed, not writing speed.
-- picchio v0.1.0 on Apple M5, 32 GB, macOS 26.5.1
```

## The three numbers

Every tok/s figure belongs to one of three lanes, and picchio never adds
them together or averages them.

Prefill is how fast the model reads your prompt. Decode is how fast it
writes the answer. Wallclock is generated tokens divided by everything,
load and warmup included, which is what your stopwatch and your gut
measure. On the machine above these are about 590, 20 and 15 in the same
run. On the CPU run below they are 25, 11 and 3. A single unlabeled
number spanning a 30x range is not a measurement, it is a rumor.

When a screenshot shows a Mac doing 500 tok/s, that is almost always
prefill. When llama-bench prints tg128, that is decode. When an app
feels slow before the first word appears, that is cold load plus
prefill, and no decode number will explain it. Before you post your next
tok/s number, it costs one minute to run this and find out which lane it
lives in.

## The number everyone posts cannot see your bottleneck

Measured here, the GPU buys about 2x on decode and about 24x on prefill
(both runs are in [examples/](examples/), 4 of 10 cpu threads on the CPU
side). Nearly every tok/s figure posted online is decode, because decode
is the one that feels like typing speed. But on consumer hardware the
pain lives mostly in prefill: it decides how long a long prompt sits
silent before the first word. Two setups can post the same decode number
while one takes ten times longer to start answering. And if your engine
quietly fell back to CPU, decode is exactly the number that will not
tell you.

## The hollow spot: silent CPU fallback

Same machine, same model, same file, forced to CPU
([examples/cpu-fallback.txt](examples/cpu-fallback.txt)):

```
model    Qwen3.5-9B-Q4_K_M.gguf, 8.95 B, 5.28 GiB, llama.cpp b9430
gpu      NOT ENGAGED: 0/33 layers on GPU
                 prefill         decode      wallclock
  pass 1      24.8 tok/s     10.7 tok/s      2.7 tok/s
  pass 2      24.9 tok/s     10.8 tok/s      2.8 tok/s
where pass 1 went (47.9 s wall, 4/10 threads, weights cached)
  load weights    4.4 s  ###.........................    9%
  prefill        30.7 s  ##################..........   64%
  decode         11.8 s  #######.....................   25%
  engine misc     0.9 s  #...........................    2%
VERDICT: SILENT CPU FALLBACK
  0 of 33 layers reached the GPU. Decode (10.8 tok/s) looks
  passable, which is how this hides. Prefill at 25 tok/s puts a
  2500 token prompt 101 s from its first word. Check -ngl.
-- picchio v0.1.0 on Apple M5, 32 GB, macOS 26.5.1
```

Look at what moved and what did not. Decode dropped 2x, which in a chat
you might shrug at. Prefill dropped 24x, and the first word of a long
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
python3 picchio.py model.gguf --explain 36
```

```
YOUR NUMBER: 36.0 tok/s -> MATCHES NOTHING MEASURED HERE
  36.0 tok/s is not within 30% of anything measured here
  (closest: decode, off by 1.8x; measured: prefill 584.2, decode
  20.0, wallclock 14.7 tok/s). Before trusting that number, ask
  which of the three rates it was, and on what hardware, quant,
  and context length.
```

That 36 is the exact number from the story above, asked against the
machine it supposedly came from. After a diagnostic run picchio caches
the rates, so later you can call `--explain` alone without rerunning.

## Ollama mode

Give picchio an ollama model tag instead of a file path and it runs the
same two passes through your local ollama server (default
`127.0.0.1:11434`, or set `OLLAMA_HOST`). You get the same three lanes,
the same first pass breakdown, and a placement check based on the memory
split ollama itself reports: how much of the model sits in GPU memory
versus CPU memory.

Real run, same weights imported into ollama
([examples/ollama-qwen35.txt](examples/ollama-qwen35.txt)):

```
model    qwen3.5:9b, 9.0 B, Q4_K_M, 5.55 GiB, ollama 0.31.1
gpu      ENGAGED: 100% of weights in GPU memory (ollama ps)
                 prefill         decode      wallclock
  pass 1     513.1 tok/s     18.5 tok/s     12.9 tok/s
  pass 2     768.6 tok/s     19.5 tok/s     16.5 tok/s
where pass 1 went (9.9 s wall)
  load weights    1.5 s  ####........................   15%
  prefill         1.5 s  ####........................   15%
  decode          6.9 s  ####################........   70%
  engine misc     0.0 s  ............................    0%
VERDICT: HEALTHY
  Ollama reports 100% of weights in GPU memory. Quote decode
  (19.5 tok/s) when you compare setups. 769 tok/s is prefill:
  reading, not writing.
-- picchio v0.1.0 on Apple M5, 32 GB, macOS 26.5.1
```

Be aware of what this mode cannot see, because ollama does not expose
it: per layer placement, device init logs, and thread configuration.
That is why llama.cpp mode is the full diagnosis and ollama mode is
measurement plus a placement check. If ollama gives no memory split at
all, picchio reports the placement as unknown instead of guessing.
These two modes are the whole scope; picchio stays one readable file.

## Is this not just llama-bench?

llama-bench is good and you should use it. It answers a different
question. It tells you how fast this machine can run this model: separate
pp and tg rates, steady state, warmup on by default. picchio tells you
what actually happened on a real run and why it felt the way it felt.

Concretely, measured on this machine, same model, same day:

| tool, config              | prompt side   | generation side | notes                     |
|---------------------------|---------------|-----------------|---------------------------|
| llama-bench, default      | pp256: 610.13 | tg64: 20.87     | backend column: BLAS,MTL  |
| llama-bench, -ngl 0 (CPU) | pp128: 30.66  | tg32: 13.25     | backend column: BLAS,MTL  |

Both rows report the same backend, because that column describes what
the binary was compiled with, not where your tokens were computed. The
20x prompt side collapse is the only visible trace of the CPU run, and
you can only read it if you already know the healthy baseline. There is
also no load time, no cold and warm split, and no interpretation; that
last part is fair, a benchmark is not supposed to have opinions.

picchio exists for the layer under the numbers: was the GPU engaged,
with the engine's own placement evidence attached, where did the first
ten seconds go, and which lane does a given number belong to.

## Measured on this machine

Apple M5, 32 GB, macOS 26.5.1, llama.cpp build 9430 and ollama 0.31.1,
roughly 730 prompt tokens and 128 generated tokens per pass. Ranges are
min to max across the recorded runs in [examples/](examples/). Every
number in this table came out of a real run on this hardware; there are
no projected or extrapolated numbers anywhere in this repo, and rows
for hardware I do not own stay empty until someone runs it there.

| config                          | prefill tok/s | decode tok/s | wallclock tok/s |
|---------------------------------|---------------|--------------|-----------------|
| Qwen3.5-9B Q4_K_M, Metal 33/33  | 584.2 - 591.5 | 18.5 - 20.3  | 11.9 - 14.7     |
| Qwen3.5-9B Q4_K_M, CPU 0/33     | 24.8 - 24.9   | 10.7 - 10.8  | 2.7 - 2.8       |
| qwen3.5:9b via ollama, 100% GPU | 513.1 - 768.6 | 18.5 - 19.5  | 12.9 - 16.5     |

Load time for the 5.28 GiB file: 3.3 s the first time it was ever read,
1.7 s after a cache flush, 0.4 s when the weights were still in the disk
cache. picchio prints a note when your pass 1 was not a true cold start,
because a cached load will flatter your first token time. One more
thing measured the hard way while building this: a large download
running in the background cut decode roughly in half on this machine,
so run picchio on a machine that is otherwise idle.

## Verdicts from other machines

I only own one computer, which is why most of this table is empty rows.
Run picchio once and paste the verdict block into an issue, even if it
says everything is fine; a boring HEALTHY on hardware I do not have is
still a data point. And if the verdict gets your machine wrong, that is
the issue I want most. Misdiagnosis reports go to the top of the pile,
because a diagnostic that misreads machines it has never met is just a
mirror with opinions.

| chip     | ram   | model, engine                      | prefill | decode | wallclock | verdict |
|----------|-------|------------------------------------|---------|--------|-----------|---------|
| Apple M5 | 32 GB | Qwen3.5-9B Q4_K_M, llama.cpp b9430 | 591.5   | 20.3   | 14.7      | HEALTHY |
| Apple M5 | 32 GB | qwen3.5:9b, ollama 0.31.1          | 768.6   | 19.5   | 16.5      | HEALTHY |
|          |       |                                    |         |        |           |         |
|          |       |                                    |         |        |           |         |
|          |       |                                    |         |        |           |         |

## What it does not do yet

The tested path is one Apple Silicon machine, llama.cpp and ollama.
Linux parsing (CUDA and Vulkan log lines, /proc hardware info) is
written but has not touched real hardware; if you run it there, I want
the verdict block either way. MLX, LM Studio and remote servers are out
of scope. Old llama.cpp builds are handled with a flag fallback ladder,
but very old builds may only get partial evidence, and picchio will say
so rather than guess. Both passes run back to back, so pass 1 is only a
true cold start if the model was not recently loaded; when the load
times give that away, the block says weights cached.

Exit codes, for scripting: 0 healthy or no evidence, 2 could not run,
3 partial offload, 4 silent CPU fallback.

## License

MIT.

<!-- TODO: footer product link pending publisher identity decision -->
