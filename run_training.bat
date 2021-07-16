REM Train the linear regression for Dask and Pandas dataframes

python train_dask.py >> train_results.txt
python train_pandas.py >> train_results.txt

PAUSE 60

SHUTDOWN /s
