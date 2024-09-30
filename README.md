# Uniqlo-price-prediction-


## How to use
Open your cmd, cd to your directory and folow these codes: 

```
git clone https://github.com/Trgtuan10/Uniqlo-price-prediction.git


cd Uniqlo-price-prediction

python -m venv .env

.env\Scripts\activate

pip install -r requirements

cd datasets

python get_datasets.py

```

## Training
Train new checkpoint
```
cd Uniqlo-price-prediction

python train_uniqlo.py --train_epoch 300 --batchsize 8 --device 0 --height 256 --width 256

```

Train pretrain or resume
```
python train_uniqlo.py --train_epoch 300 --batchsize 8 --device 0 --height 256 --width 256 --use_pretrain True --pretrain_model 'path/to/your/checkpoint.pt'
```

## Test your clothes 
My demo: [Uniqlo-information-prediction](https://mt-uniqlo.streamlit.app/)
