from model import bayesian_uncertainty

def __run__():
    # Initialize the model
    model = bayesian_uncertainty()
    # Import the dataset:
    model.import_dataset(target='covid_20')

if __name__ == "__main__":
    __run__()
