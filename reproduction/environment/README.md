# Environment records

`training.fau.snapshot.txt` and `evaluation.fau.snapshot.txt` are the dependency files saved in the FAU project at release preparation time. `../evaluation/requirements.snapshot.txt` is the saved local evaluation requirements. These are provenance snapshots, not a claim that all historical seeds used exactly the same environment. Use a Linux CUDA environment for the bitsandbytes LAMB training path; configure a compatible PyTorch/CUDA stack before installing dependencies. The small `../../requirements.txt` lists the direct training dependencies.

No historical training was rerun to validate these snapshots during publication. The original generator/readability textstat version and bootstrap RNG state were not retained. Their replacement analysis tools are labelled as reconstructions.
