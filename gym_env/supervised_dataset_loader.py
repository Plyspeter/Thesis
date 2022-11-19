from sklearn import datasets
import hedwig

# This could be improved by not having to load the data
# every time a new supervised env is created

class SupervisedDatasetLoader:

    def __init__(self, dataset_name : str):
        if dataset_name == 'Iris':
            dataset = datasets.load_iris()
            self.data = list(dataset.data)
            self.target = list(dataset.target)

            # Remove data for validation set
            del self.data[0:5]
            del self.data[50:55]
            del self.data[100:105]

            del self.target[0:5]
            del self.target[50:55]
            del self.target[100:105]

            # Create validation set
            self.data_val = list(dataset.data[0:5]) + list(dataset.data[50:55]) + list(dataset.data[100:105])
            self.target_val = [0] * 5 + [1] * 5 + [2] * 5

            self.input_size = 4
            self.output_size = 3
        else:
            hedwig.error(f"Failed to load supervised env: {dataset_name}")
            raise Exception(f"Supervised ENV is not registered: {dataset_name}")