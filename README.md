# `regi_py`

This repo contains a C++ implementation of the game mechanics of the card game
[`regicide`](https://badgersfrommars.com/en-us/pages/learn-to-play-regicide), 
with Python wrappers (via `pybind11`), and a WASM wrapper.

Try the WASM version of the game [here](https://ahgamut.github.io/regi-py): once
loaded, the game (including neural-net bots) should run fully in the browser,
without requiring any connection to a server.

![](./webdriver/wasm/regi-runtime.png)

## Installation

Install this package via Python's `pip` tool on the command line:

```sh
# activate anaconda/uv/whatever for Python env
git clone https://github.com/ahgamut/regi-py
cd regi-py
python3 -m pip install -e .
```


## Building and testing

The package ships a C++ extension (`regi_py.core`) that must be compiled for your
interpreter. Build it in-place and run the test suite with:

```sh
# from a fresh virtualenv
python -m pip install --upgrade pip setuptools wheel
python -m pip install "pybind11>=3"
python -m pip install -e .[dev] --no-build-isolation   # builds the extension + test deps
pytest -q
```

Re-run `pip install -e . --no-build-isolation` after any change to the C++ sources
under `src/regi_py/core/` to rebuild the extension before testing.

```sh
python -m pytest webdriver/tests   # NOT bare `pytest webdriver/tests`
```

## Viewing a basic simulation

Run the `driver.py` to see a basic command-line simulation of the game, where
each player randomly selects a valid move to play.

```sh
# install the package first
python driver.py
```

To use custom strategies available in the package, specify them with
`--add-bot`:

```sh
python driver.py --help # view options for bots
python driver.py --add-bot damage --add-bot preserve
```

## Playing with a bot

To play locally, run `webdriver/driver.py` to play a 2-player game on your local server:

```sh
# install the package first
cd webdriver
python driver.py \
    -n 1 \ # one human player needs to connect
    --add-bot damage # add the 'damage' bot
# go to http://localhost:8888 in your browser
```

![](./webdriver/regi-runtime.png)

You can also build the WASM version and run it locally. To use the neural nets
locally, you'll need to train them by yourself.

## Adding your own strategies

Subclass the `BaseStrategy` class with your own implementations that select what
attack/defense moves to make.
