import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

X = pd.read_csv("test_data.csv", header=0).values
y = pd.read_csv("test_data_result.csv", header=0).values

model_names = ['dt_model', 'lr_model', 'nn_model', 'rf_model','svc_model', 'stacked_model']
models = []
for i in model_names:
    path = './models/' + i + '.pkl'
    with open(path, 'rb') as f:
        models.append(pickle.load(f))

new_pred1 = models[0].predict(X)
print(new_pred1)

new_pred2 = models[1].predict(X)
accuracy = accuracy_score(y, new_pred2)
cm = confusion_matrix(y, new_pred2)
print(f"Model: {model_names[1]}")
print("Accuracy: {:.6f}".format(accuracy))
print(cm)

new_pred3 = models[2].predict(X)
new_pred3 = (new_pred3 > 0.5)
accuracy = accuracy_score(y, new_pred3)
cm = confusion_matrix(y, new_pred3)
print(f"Model: {model_names[2]}")
print("Accuracy: {:.6f}".format(accuracy))
print(cm)

new_pred4 = models[3].predict(X)
accuracy = accuracy_score(y, new_pred4)
cm = confusion_matrix(y, new_pred4)
print(f"Model: {model_names[3]}")
print("Accuracy: {:.6f}".format(accuracy))
print(cm)

new_pred5 = models[4].predict(X)
accuracy = accuracy_score(y, new_pred5)
cm = confusion_matrix(y, new_pred5)
print(f"Model: {model_names[4]}")
print("Accuracy: {:.6f}".format(accuracy))
print(cm)

new_X_test = np.column_stack((new_pred1, new_pred2, new_pred3, new_pred4, new_pred5))
new_pred6 = models[5].predict(new_X_test)

accuracy = accuracy_score(y, new_pred6)
cm = confusion_matrix(y, new_pred6)
print("Accuracy: {:.6f}".format(accuracy))
print(cm)
