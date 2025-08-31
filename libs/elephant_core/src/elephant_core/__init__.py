try:
    from ._version import __version__  # ビルド/インストール後はここが使われる
except Exception:  # 開発直後など
    from importlib.metadata import version as _v

    try:
        __version__ = _v("elephant-core")
    except Exception:
        __version__ = "0+unknown"
