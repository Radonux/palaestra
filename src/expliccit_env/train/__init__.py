"""Training net1 (path beliefs) and net2 (the controller) from step data.

    spec.py     what to train (TrainSpec), parsed once
    data.py     steps.parquet shards -> arrays, split by path, never by row
    models.py   architectures by name, and the in-graph input stage
    fit.py      net1 supervised, net2 offline RL on frozen net1 outputs
    export.py   one ONNX graph per policy, under the sage-v1 contract

Needs the ``train`` extra (PyTorch). Architectures and algorithms are chosen
by name in the spec; a new design is a new registry entry, not a new
pipeline.
"""
