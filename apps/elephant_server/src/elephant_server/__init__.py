try:
    from ._version import __version__
except Exception:
    from importlib.metadata import version as _v

    try:
        __version__ = _v("elephant-server")
    except Exception:
        __version__ = "0+unknown"
