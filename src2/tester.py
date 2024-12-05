import matplotlib.pyplot as plt

# Models and their accuracies after 10 epochs
models = ["CNN", "LSTM", "Xi-Net Classifier", " Base Transformer"]
accuracies = [98, 90, 25, 40]  # Accuracies in percentage

# Plotting the bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])

# Adding accuracy labels on top of each bar
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5,
             f'{accuracy}%', ha='center', va='bottom', color='white', fontsize=12)

# Setting the title and labels
plt.title('Model Accuracies After 10 Epochs on MIT-BIH Arrhythmia Dataset')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  # Setting y-axis limit to 100%

# Displaying the plot
plt.show()
