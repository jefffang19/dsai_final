# DSAI HW4 (final)

組員: 陳香君 方郁文

## Instacart Market Basket Analysis
https://www.kaggle.com/c/instacart-market-basket-analysis
<br>
Our report can be viewed and download here<br>
https://docs.google.com/presentation/d/1UjyoupZZajfqWIy9ncTT5KUUOVqBp1NS1yANfAtjLeE/edit?usp=sharing

## How to reproduce this homework
`python version == 3.6.13`
1. download input data from 
```
https://drive.google.com/file/d/1Aw2Mg7BpVG5P7lw5_ABJsRku8AsAkJ7p/view?usp=sharing
```
2. put `uxp.csv` into `data/`
3. install requirements with
```
pip install -r requirements.txt
```
4. run training with
```
python run_model.py
```
5. results are saved as submission.csv


Our Codes uses CPU with multi-threading<br>

My Specs<br>
```
Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
4 cores 8 threads
```
time took to train xgboost classifier "296.1159" seconds<br>

## notebook explains
- `process_data.ipynb` is the file we use to process the input data
- `MF.ipynb` we tried Matrix Factorization, but did not work as well as xgboost
- `xgboost.ipynb` the code we use to test performance of xgboost, `run_model.py` is based on this notebook
