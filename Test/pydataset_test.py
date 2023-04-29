from pydataset import data

# Load a dataset from the library into a pandas DataFrame
df = data('Titanic')

# print all the datasets in the library
print(data()[100:120])