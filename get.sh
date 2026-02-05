#!/bin/bash
for layer in {0..31}; do
    uv run get_activation.py --layer $layer
done