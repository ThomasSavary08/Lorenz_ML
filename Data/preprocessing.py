# Libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
seq_length = 200

# Load raw data
trainCLV_raw, testCLV_raw = np.load('./../CLVs/trainCLV.npy'), np.load('./../CLVs/testCLV.npy')
trainTraj, testTraj = np.load('./../CLVs/trainTraj.npy'), np.load('./../CLVs/testTraj.npy')

# Compute angles between CLV
n_train, dim, _ = trainCLV_raw.shape
n_test, _, _ = testCLV_raw.shape
angles_train, angles_test = np.zeros((n_train, dim)), np.zeros((n_test, dim))

print("Compute angles between CLVs.")
for i in range(n_train):
    angles_train[i,0] = np.arccos(np.dot(trainCLV_raw[i,:,0], trainCLV_raw[i,:,1]))
    angles_train[i,1] = np.arccos(np.dot(trainCLV_raw[i,:,0], trainCLV_raw[i,:,2]))
    angles_train[i,2] = np.arccos(np.dot(trainCLV_raw[i,:,1], trainCLV_raw[i,:,2]))
    if (i < n_test):
        angles_test[i,0] = np.arccos(np.dot(testCLV_raw[i,:,0], testCLV_raw[i,:,1]))
        angles_test[i,1] = np.arccos(np.dot(testCLV_raw[i,:,0], testCLV_raw[i,:,2]))
        angles_test[i,2] = np.arccos(np.dot(testCLV_raw[i,:,1], testCLV_raw[i,:,2]))
print("OK.")
print("")

# Find wing changes in the trajectory
def which_wing(x, y):
    if ((x > 0) and (y > 0)):
        return 1
    elif ((x < 0) and (y < 0)):
        return -1
    else:
        return 0

print("Find wing changes in training trajectory.")
ind_start_train = seq_length - 1
while which_wing(trainTraj[ind_start_train, 0], trainTraj[ind_start_train, 1]) == 0:
    ind_start_train += 1
current_wing = which_wing(trainTraj[ind_start_train, 0], trainTraj[ind_start_train, 1])
if current_wing != which_wing(trainTraj[ind_start_train - 1, 0], trainTraj[ind_start_train - 1, 1]):
    changes_train = [1]
else:
    changes_train = [0]
for i in range(ind_start_train + 1, n_train):
    computed_wing = which_wing(trainTraj[i,0], trainTraj[i,1])
    if ((computed_wing == current_wing) or (computed_wing == 0)):
        changes_train.append(0)
    else:
        current_wing = computed_wing
        changes_train.append(1)
print("OK.")
print("")

print("Find wing changes in test trajectory.")
ind_start_test = seq_length - 1
while which_wing(testTraj[ind_start_test, 0], testTraj[ind_start_test, 1]) == 0:
    ind_start_test += 1
current_wing = which_wing(testTraj[ind_start_test, 0], testTraj[ind_start_test, 1])
if current_wing != which_wing(testTraj[ind_start_test - 1, 0], testTraj[ind_start_test - 1, 1]):
    changes_test = [1]
else:
    changes_test = [0]
for i in range(ind_start_test + 1, n_test):
    computed_wing = which_wing(testTraj[i,0], testTraj[i,1])
    if ((computed_wing == current_wing) or (computed_wing == 0)):
        changes_test.append(0)
    else:
        current_wing = computed_wing
        changes_test.append(1)
print("OK.")
print("")

# Find last transition in each trajectory and compute the number of time steps before the next transition
print("Find the number of time steps before next transition in traning trajectory.")
ind_end_train = changes_train[::-1].index(1)
ind_end_train = len(changes_train) - ind_end_train - 1
before_changes_train = [0]
nb = 0
for i in range(ind_end_train - 1, -1, -1):
    if changes_train[i] == 1:
        nb = 0
    else:
        nb +=  1
    before_changes_train.append(nb)
before_changes_train = before_changes_train[::-1]
changes_train = changes_train[:ind_end_train + 1]
print("OK.")
print("")

print("Find the number of time steps before next transition in test trajectory.")
ind_end_test = changes_test[::-1].index(1)
ind_end_test = len(changes_test) - ind_end_test - 1
before_changes_test = [0]
nb = 0
for i in range(ind_end_test - 1, -1, -1):
    if changes_test[i] == 1:
        nb = 0
    else:
        nb +=  1
    before_changes_test.append(nb)
before_changes_test = before_changes_test[::-1]
changes_test = changes_test[:ind_end_test + 1]
print("OK.")
print("")

# Convert the number of time steps to labels (classification)
print("Compute labels for classification.")
labels_train, labels_test = [], []
for i in range(len(before_changes_train)):
    if before_changes_train[i] < 50:
        labels_train.append(0)
    elif ((before_changes_train[i] >= 50) and (before_changes_train[i] < 150)):
        labels_train.append(1)
    else:
        labels_train.append(2)
    if i < len(before_changes_test):
        if before_changes_test[i] < 50:
            labels_test.append(0)
        elif ((before_changes_test[i] >= 50) and (before_changes_test[i] < 150)):
            labels_test.append(1)
        else:
            labels_test.append(2)
print('OK.')
print("")

# Create datasets
print("Create and save datasets.")
x_train, x_test = [], []
for i in range(ind_start_train, ind_start_train + len(labels_train)):
    x_train.append(trainTraj[i - seq_length + 1:i + 1,:])
for i in range(ind_start_test, ind_start_test + len(labels_test)):
    x_test.append(testTraj[i - seq_length + 1:i + 1,:])
x_train, y_train = np.asarray(x_train), np.asarray(labels_train)
x_test, y_test = np.asarray(x_test), np.asarray(labels_test)
np.save('./x_train.npy', x_train)
np.save('./y_train.npy', y_train)
np.save('./x_test.npy', x_test)
np.save('./y_test.npy', y_test)
print("OK.")