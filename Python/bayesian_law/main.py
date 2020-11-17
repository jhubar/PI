from model import seir

def __run__():
    # Initialize the model
    model = seir()
    # Import the dataset:
    model.import_dataset(target='covid_20')
    model.fit()
    predictions = model.predict(300)
    model.plot_predict(predictions, args='predict no_S')
    model.print_final_value()
    model.saveJson()

if __name__ == "__main__":
    __run__()
