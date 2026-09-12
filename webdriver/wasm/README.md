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

Open `http://localhost:8000`, pick the player count / bot net (or **Spectate** to
watch bots), and play seat 0.

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
- Only **ADZ** nets (`adzpool`, `adzmulti`) are wired as bots so far; the AZ
  card-space nets (`basic`, `cardtx`, …) would need the `combomap` bijection
  shipped as JSON plus the keepyness defense fallback.
- Bots play **Direct-net** (search-free argmax). MCTS Explorer is a later phase.
