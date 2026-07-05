# =============================================================================
# SUFLECA
#
# SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
# SPDX-License-Identifier: Apache-2.0
#
# File: sufleca/model/__init__.py
#
# Copyright © 2023-2026 University of Luxembourg
# Developed by Saad Ejaz at SnT/ARG.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =============================================================================

from .config import SUFLECAConfig, sufleca_config_from_checkpoint
from .extractor import SUFLECAFeatureExtractor

__all__ = [
    "SUFLECAConfig",
    "SUFLECAFeatureExtractor",
    "sufleca_config_from_checkpoint",
]
