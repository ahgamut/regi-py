"""Back-compat CLI shim: rowify msgpack MCTS NodeInfo records into a CSV.

The implementation now lives in ``logs2df.py`` (shared with the JSON path);
this preserves the ``python bin2df.py -i ... -o ...`` command line.  ``msgpack``
is imported lazily by the msgpack source, only when a file is actually read.
"""

from logs2df import main, MsgpackSource

if __name__ == "__main__":
    main(MsgpackSource())
