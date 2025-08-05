# from .service import run_agent  # noqa
# __all__ = ["run_agent"]
try:
    from .service import run_agent  # noqa: F401
    __all__ = ["run_agent"]
except (ImportError, AttributeError):
    __all__ = []