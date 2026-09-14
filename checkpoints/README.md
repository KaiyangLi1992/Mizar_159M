# Released checkpoint

Download `checkpoints/final/stage3_q4_seed20260905.ckpt` with
`python pipeline/download_assets.py`. It is the full joint strongAC + AVQA Q4
model at update 200, containing encoder, mapper and decoder weights.

S1/S2 intermediate weights and other final seeds are not uploaded. The source
code and five data manifests support reproducing those runs. The released
checkpoint is a plain PyTorch tensor state dictionary, suitable for strict
loading with the supplied architecture and `weights_only=True`.
