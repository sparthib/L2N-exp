import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras import layers
from proglearn.network import LifelongClassificationNetwork
from proglearn.transformers import NeuralClassificationTransformer
from proglearn.voters import KNNClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage

from sklearn.model_selection import train_test_split

def run_experiment(
    x_data, y_data, num_tasks, num_points_per_task, model="dnn", reps=100
):
    """Runs the FTE/BTE experiment.
    Referenced Chenyu's code, with modifications to adjust the number of tasks.
    """

    # initialize list for storing results
    accuracies_across_tasks = []

    # format data
    if model == "dnn": 
        x = x_data
        y = y_data

    # get y values per task
    unique_y = np.unique(y_data)
    ys_by_task = unique_y.reshape(num_tasks, int(len(unique_y) / num_tasks))

    # run experiment over all reps
    for rep in range(reps):
        # print('Starting rep', rep)

        # for each task
        for task in range(num_tasks):

            # initialize progressive learner
            if model == 'dnn':

                network = keras.Sequential()
                network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(x)))
                network.add(layers.BatchNormalization())
                network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
                network.add(layers.BatchNormalization())
                network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
                network.add(layers.BatchNormalization())
                network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
                network.add(layers.BatchNormalization())
                network.add(layers.Conv2D(filters=254, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))

                network.add(layers.Flatten())
                network.add(layers.BatchNormalization())
                network.add(layers.Dense(2000, activation='relu'))
                network.add(layers.BatchNormalization())
                network.add(layers.Dense(2000, activation='relu'))
                network.add(layers.BatchNormalization())
                network.add(layers.Dense(units=10, activation = 'softmax'))

                learner = LifelongClassificationNetwork(network= network)


            # get train/test data (train = num_points_per_task)
            index = np.where(np.in1d(y, ys_by_task[task]))
            x_task0 = x[index]
            y_task0 = y[index]
            train_x_task0, test_x_task0, train_y_task0, test_y_task0 = train_test_split(
                x_task0, y_task0, test_size=0.25
            )
            train_x_task0 = train_x_task0[:num_points_per_task]
            train_y_task0 = train_y_task0[:num_points_per_task]

            # feed to learner and predict on single task
            learner.add_task(train_x_task0, train_y_task0)
            task_0_predictions = learner.predict(test_x_task0, task_id=0)
            accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

            # evaluate for other tasks
            for other_task in range(num_tasks):

                if other_task == task:
                    pass

                else:

                    # get train/test data (train = num_points_per_task)
                    index = np.random.choice(
                        np.where(np.in1d(y, ys_by_task[other_task]))[0],
                        num_points_per_task,
                        replace=False,
                    )
                    train_x = x[index]
                    train_y = y[index]

                    # add transformer from other tasks
                    learner.add_task(train_x, train_y)

                # predict on current task using other tasks
                prev_task_predictions = learner.predict(test_x_task0, task_id=0)
                accuracies_across_tasks.append(
                    np.mean(prev_task_predictions == test_y_task0)
                )

    # average results
    accuracy_all_task = np.array(accuracies_across_tasks).reshape((reps, -1))
    accuracy_all_task = np.mean(accuracy_all_task, axis=0)

    return accuracy_all_task



