# Neural State Space Machine (SSM) for Speech Enhancement

This is the implementation code repo for paper "A Multi-dimensional Deep Structured State Space Approach to
Speech Enhancement Using Small-footprint Models".

[Paper](https://arxiv.org/pdf/2306.00331.pdf)

<img src="https://github.com/Kuray107/S4ND-U-Net_speech_enhancement/blob/main/s4se_is23.png" width="500">

Run and Print the Loss
```
cd S4ND-U-Net_speech_enhancement/model
python S4ND-U-Net.py

```

### ❗ UPDATE ❗

This repo is a MODIFIED version of the original repo. In short, the new version supports updating the state for the S4ND model that allows eliminating the following [code line](https://github.com/Kuray107/S4ND-U-Net_speech_enhancement/blob/main/model/DSSM_modules/s4nd.py#L226):
```
assert state is None, f"state not currently supported in S4ND"
```
More specifically, in this implementation, the last dimension is assumed to be _time_, so the corresponding S4 kernel is causal and supports recurrent inference.

In this case, setting `bidirectional=True` affects only the remaining (_spatial_) S4 kernels.


Run tests
```
pytest -q test_inference.py

```

**TODO**: implement recurrent inference for the full `S4ND_U_Net` model.

### Reference

```bib
@article{ku2023multi,
  title={A Multi-dimensional Deep Structured State Space Approach to Speech Enhancement Using Small-footprint Models},
  author={Ku, Pin-Jui and Yang, Chao-Han Huck and Siniscalchi, Sabato Marco and Lee, Chin-Hui},
  journal={arXiv preprint arXiv:2306.00331},
  year={2023}
}
```
