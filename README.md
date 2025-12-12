# hiplearn
Codes and notes on AMD HIP


Dev system specification (bare metal installation, except python venv):
 * ROCm 7.1 
 * Ubuntu 24.04 LTS with kernel 6.8 generic
 * PyTorch nightly build for ROCm 7.1 (as of Dec 2025)
 * CPU/GPU: 7900X3D/7800XT

I will probably migrate test/running env to rocm/pytorch docker soon (I used to have it until my trial to upgrade ROCm to 7.1 screwed up my ubuntu 22.04 installation completely)
