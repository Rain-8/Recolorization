# Recolorization

## Install MiniConda
### For MacOS - 
```
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
Make sure that you are using the conda interpretor, if you're on an IDE like VSCode. (macos shortcut - Cmd + Shift + P)
```
conda create -n myenv python=3.12
conda activate myenv
conda deactivate myenv # when you want to exit
pip install -r requirements.txt
```

## Training Setup
### For getting the dataset
```
dvc pull datasets/processed_palettenet_data_sample_v4
cd src/custom_model
python data.py # to visualize the data
```

### For training on GPU
```
# ensure you have the dataset
cd src/custom_model
./train_gpu.sh
```


## Streamlit App Setup

Download the model in `deployments/streamlit_app`

```
pip install -r requirements_deploy.txt
pip install watchdog
cd deployments/streamlit_app
streamlit run streamlit_app.py
```
