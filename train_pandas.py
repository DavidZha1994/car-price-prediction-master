import dask.dataframe as dd
from dask.distributed import Client
from sklearn.linear_model import LinearRegression
import joblib
import datetime
import time

def main():
    # Vary the number of jobs
    num_jobs = [1, 2, 3, 4, 5, 6]

    used_cars_hdf = 'used_cars_preprocessed.hdf'
    data = dd.read_hdf(used_cars_hdf, '/data-*')

    client = Client()

    print('Results for Pandas:')

    for i in range(3):
        print(f'\nIteration {i + 1}:')

        for n in num_jobs:
            # Create a model for linear regression
            model = LinearRegression()

            # Start time measurement and include computations
            start = time.time()

            # Compute to pandas dataframe
            computed_data = data.compute()

            # Get training data and true prices and
            # convert them to Dask arrays
            X = computed_data.drop(labels=['price'], axis=1)
            y = computed_data['price']

            # Train the model
            with joblib.parallel_backend('dask', n_jobs=n):
                model.fit(X, y)

            duration = time.time() - start
            print(f'Duration (n = {n}): {datetime.timedelta(seconds=duration)}')


if __name__ == '__main__':
    main()
