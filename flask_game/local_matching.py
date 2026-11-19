"""Backward-compatible import surface for the shared matcher.

The implementation lives in the project-root ``bioguard_matching`` module so
the master notebook and Flask import exactly the same code.
"""
from bioguard_matching import *  # noqa: F401,F403
