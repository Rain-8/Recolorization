# Recolorization

## Install MiniConda
### For MacOS - 
```
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
Make sure that you are using the conda interpretor, if you're on an IDE like VSCode.
```
conda create -n myenv python=3.12
conda activate myenv
conda deactivate myenv # when you want to exit
```
```
pip install -r requirements.txt # if works
```
```
python model.py
pip install streamlit
pip install watchdog
pip install dvc
```
