## set up VISTA 

the latest FlagEmbedding repo doesn't support VISTA anymore so we use a old version.
```
git clone --no-checkout https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
git checkout 5c9260277977f8f8e256e56a8e12387552693af9
pip install -e .
pip install torchvision timm einops ftfy
```
download the vision tower for bge-base
```
wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth?download=true
```
rename it to `visualized_base_en_V1.5.pth`

download the vision tower for bge-m3
```
wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth?download=true
```
rename it to `visualized_m3.pth`