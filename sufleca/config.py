# =============================================================================
# SUFLECA
#
# SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
# SPDX-License-Identifier: Apache-2.0
#
# File: sufleca/config.py
#
# Copyright © 2023-2026 University of Luxembourg
# Developed by Saad Ejaz at SnT/ARG.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =============================================================================

"""Read dotted-path values from dict or attribute-namespace (e.g. munch) configs."""
from __future__ import annotations

_MISSING = object()


def get_config_value(config, path: str, default=_MISSING):
    """Return ``config.<path>`` where ``path`` is dotted, e.g. ``"ransac.pct_multiplier"``.

    Works on nested dicts and attribute namespaces. If a segment is missing,
    returns ``default`` when one is given, otherwise raises ``ValueError``. With
    no ``default`` a resolved value of ``None`` is rejected the same way.
    """
    node = config
    for part in path.split("."):
        if isinstance(node, dict):
            missing = part not in node
            node = node.get(part)
        else:
            missing = not hasattr(node, part)
            node = getattr(node, part, None)
        if missing:
            if default is _MISSING:
                raise ValueError(f"Missing config field '{path}'")
            return default
    if node is None and default is _MISSING:
        raise ValueError(f"Config field '{path}' is None")
    return node
