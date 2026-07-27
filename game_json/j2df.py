"""Back-compat CLI shim: rowify JSON event logs into a CSV.

The implementation now lives in ``logs2df.py`` (shared with the msgpack path);
this preserves the ``python j2df.py -i ... -o ...`` command line.
"""
from logs2df import main, JsonSource

if __name__ == "__main__":
    main(JsonSource())
