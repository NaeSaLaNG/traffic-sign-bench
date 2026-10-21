"""Policy lab: an LLM agent writes CaRL-based driving policies in a sandbox.

The judge side (this package) owns the simulator, sign rules and scoring; the
agent only sees the exported sandbox (``export_sandbox``) and aggregate
metrics returned by ``judge``. See README.md.
"""
