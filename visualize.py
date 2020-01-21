import pickle
import numpy as np 
import matplotlib.pyplot as plt


training_data = pickle.load(open('./save/training_data.pickle', 'rb'))

# epochs = np.array([time_step[0] for time_step in training_data])
train_losses = np.array([time_step[1] for time_step in training_data])
val_losses = np.array([time_step[2].cpu().numpy() for time_step in training_data])
val_accuracies = np.array([time_step[3].cpu().numpy() for time_step in training_data])
epochs = np.linspace(0, 49, num=250)

plt.figure()
plt.title('epochs vs. loss')
plt.plot(epochs, train_losses, label='train_losses')
plt.plot(epochs, val_losses, label='val_losses')
plt.xlabel('epochs')
plt.ylabel('nll_loss')
plt.legend()
plt.savefig('./save/epochs_vs_loss.png', format='png')

plt.figure()
plt.title('epochs vs. accuracy')
plt.plot(epochs, val_accuracies)
plt.xlabel('epochs')
plt.ylabel('val_accuracy')
plt.savefig('./save/epochs_vs_accuracy.png', format='png')

