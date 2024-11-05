# Recolorization

## Install MiniConda
### For MacOS - 
```
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

```
conda create -n myenv python=3.12
```
```
pip install -r requirements.txt
```
```
python model.py
```
