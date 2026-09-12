# Client-side Regicide (WASM + ONNX)

The whole game runs in the browser: the C++ engine compiled to WebAssembly plus
net bots run under onnxruntime-web. There is **no server game state and no
per-move round-trip** — the static host just ships blobs (the `.wasm` engine, the
`.onnx` net weights, and this JS).

## Layout

| file | role |
|------|------|
| `CMakeLists.txt`, `embind.cc` | WASM build of the core engine + Embind bindings |
| `dist/regicore.{mjs,wasm}` | built engine module (git-ignored) |
| `dist/<net>.onnx`, `dist/<net>.io.json` | exported net + its IO contract (git-ignored) |
| `adz_bot.mjs` | `NetBot` — featurize + onnxruntime forward + argmax (ADZ Direct) |
| `az_bot.mjs` | `AZBot` — card-space Direct bot (combomap grid / keepy defense) |
| `load_bot.mjs` | `loadBot`/`buildBot` — pick `NetBot`/`AZBot`/`ExplorerBot` from the contract + iters |
| `tables/combomap.json` | AZ combo bitwise → `(loc, played-status)` grid cell map |
| `tables/presets_{2,3,4}p.json` | committed starter openings (phase strings) the menu offers |
| `gen_presets.mjs` | regenerate the preset openings from the WASM engine (node) |
| `phase_expander.mjs` | `PhaseExpander` — step to the next decision node (MCTS child gen) |
| `mcts.mjs` | `MCTSNode` + `ExplorerBot` — net-guided search (both paradigms) |
| `game_driver.mjs` | `GameDriver` — the browser game loop (`prepare()`/`commit()`) |
| `app.mjs`, `index.html`, `app.css` | the UI |
| `smoke*.mjs` | node smoke tests (engine, featurizer, bot feeds, full driver) |
| `gen_golden.py`, `check_golden.mjs` | JS-vs-Python Direct-net index parity check |

## Build & run

From the repo root, in an environment with the Emscripten SDK + CMake + Ninja +
Node, and (for the ONNX export) the torch env:

```bash
# 1. build the WASM engine
cd webdriver/wasm
emcmake cmake -G Ninja -B build && cmake --build build      # -> dist/regicore.{mjs,wasm}

# 2. export the net(s) to ONNX (torch env; run from the repo root)
cd ../..
python -m trainers.export_onnx export --net adzpool \
    --weights weights/best_adzpool.pt --out webdriver/wasm/dist/adzpool.onnx --verify
python -m trainers.export_onnx export --net adzmulti \
    --weights weights/best_adzmulti.pt --out webdriver/wasm/dist/adzmulti.onnx --verify

# 3. get onnxruntime-web (pick ONE) and serve
cd webdriver/wasm
npm install                                   # installs into node_modules/ (the default)
python3 -m http.server 8000                   # serve (no npm needed for this step)
```

Export any nets you want to offer as bots (`adzpool`/`adzmulti` are ADZ;
`basic`/`percardmlp`/`cardtx`/`mixer`/`movetoken` are AZ — the AZ nets also need
`tables/combomap.json`, which is committed). Open `http://localhost:8000`, set the
player count, pick each opponent's net AND its search depth (Direct, or an MCTS
Explorer at 16/32/64/128 iterations), choose the **opening deal** (a random deal, or
one of the committed presets for that player count), then play your (shuffled) seat.

The presets in `tables/presets_{2,3,4}p.json` are fixed opening deals (replayed via
`init_string`) so a known scenario can be re-played; regenerate them with
`node gen_presets.mjs` (they're committed with `git add -f`, past the global `*.json`
ignore, like `combomap.json`).

### Running without npm

Serving never needs npm — any static server works (`python3 -m http.server 8000`).
Only onnxruntime-web has to come from somewhere; set `ORT_DIST` at the top of
`app.mjs` to one of:

- **CDN** (needs network once, then browser-cached):
  `const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';`
- **Vendored** (fully offline): download the two files with curl and point at them:
  ```bash
  mkdir -p vendor && cd vendor
  base=https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist
  curl -LO $base/ort.wasm.bundle.min.mjs
  curl -LO $base/ort-wasm-simd-threaded.wasm
  cd .. # then: const ORT_DIST = './vendor/';
  ```

The ESM loads its `.wasm` sidecar (~10 MB) from `ORT_DIST` (set as
`ort.env.wasm.wasmPaths`), so the `.mjs` and the `.wasm` must sit in the same dir.

## Tests

```bash
npm run smoke     # engine + featurizer + bot-feed + full-driver smoke (node)
# JS-vs-Python Direct-net parity (needs a fixture from the torch env):
python -m webdriver.wasm.gen_golden --net adzpool \
    --weights weights/best_adzpool.pt --out webdriver/wasm/golden/adzpool.json
npm run golden
```

## Notes

- **Single-threaded** onnxruntime-web (no SharedArrayBuffer), so no COOP/COEP
  headers are required — a plain static host (or GitHub Pages) works.
- Both paradigms are wired as bots: **ADZ** (`adzpool`, `adzmulti`, candidate
  scoring) and **AZ** (`basic`, `percardmlp`, `cardtx`, `mixer`, `movetoken`,
  card-space via the `combomap` grid + keepyness defense fallback). `attntrunk` is
  omitted.
- Bots play either **Direct-net** (search-free argmax) or an **MCTS Explorer**
  (net-guided search, ~iters+1 forward passes/move; picked per bot, 16–128 iters).
- When a game ends, the result overlay shows summary stats (royals cleared / damage
  dealt / moves) and its opening deal; **Review board** dismisses it to inspect the
  finished board, and a floating **Show result** button brings the summary back.
