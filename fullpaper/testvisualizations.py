import matplotlib.pyplot as plt


import json

def test_visualizations():

    # Read the text file back into a dictionary
    with open('my_dict.txt', 'r') as file:
        data = json.load(file)

    #Extracting accuracy data
    window_sizes = list(data.keys())
    models = list(data[1].keys())
    accuracy_data = {model: [data[window_size][model]['accuracy'] for window_size in window_sizes] for model in models}

    # Plotting
    plt.figure(figsize=(10, 6))
    for model in models:
        # if model != "kmeans":
        plt.plot(window_sizes, accuracy_data[model], marker='o', label=model)

    plt.xlabel('Window Size')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Window Size')
    plt.xticks(window_sizes)
    plt.legend()
    plt.grid(True)
    plt.show()


    # Extracting accuracy data
    window_sizes = list(data.keys())
    models = list(data[1].keys())
    accuracy_data = {model: [data[window_size][model]['accuracy'] for window_size in window_sizes] for model in models}

    # Calculating the margin increase in accuracy per model over window size change intervals
    margin_data = {model: [accuracy_data[model][i] - accuracy_data[model][i-1] for i in range(1, len(window_sizes))] for model in models}

    # Plotting the margin increase graph
    plt.figure(figsize=(10, 6))
    for model in models:
        # if model != "kmeans":
        plt.plot(window_sizes[1:], margin_data[model], marker='o', label=model)

    plt.xlabel('Window Size')
    plt.ylabel('Margin Increase in Accuracy')
    plt.title('Margin Increase in Model Accuracy per Window Size Interval')
    plt.xticks(window_sizes[1:])
    plt.legend()
    plt.grid(True)
    plt.show()