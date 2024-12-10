import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import time
import joblib
import copy


Train = False
# Load dataset
data = np.load('sv_history.npy')
data_pairs = []
for i in range(len(data)):
    data_i = data[i]
    j = int(len(data_i) / 25)
    for k in range(j):
        data_pairs.append((data_i[k*15:k*15+15].reshape(1, 15*5), data_i[k*15+16:k*15+26].reshape(1, 10*5)))
# rearrange the order of the data pairs
np.random.shuffle(data_pairs)
# Preprocess data
history_data = np.array([x[0] for x in data_pairs])
future_data = np.array([x[1] for x in data_pairs])

# show some data pairs
# plt.figure(figsize=(4, 3))
# for i in range(50):
#     # k = np.random.randint(0, history_data.shape[0])
#     k = i
#     # draw the history and future trajectory with legend
#     for j in range(15):
#         if k == 19 and j == 0:
#             plt.plot(history_data[k][0][j*5], history_data[k][0][j*5+1], 'ro', label='history position', markersize=2)
#         else:
#             plt.plot(history_data[k][0][j*5], history_data[k][0][j*5+1], 'ro', markersize=2)
#     for j in range(10):
#         if k == 19 and j == 0:
#             plt.plot(future_data[k][0][j*5], future_data[k][0][j*5+1], 'bo', label='future position', markersize=2)
#         else:
#             plt.plot(future_data[k][0][j*5], future_data[k][0][j*5+1], 'bo', markersize=2)
# plt.axis('equal')
# # the x axis is the x axis in the map frame, the y axis is the y axis in the map frame
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.legend()



# Normalize and centralize the first 2 elements of the data
for i in range(history_data.shape[0]):
    extract_val = copy.deepcopy(history_data[i][0][0:2])
    for j in range(15):
        history_data[i][0][j*5:j*5+2] = (history_data[i][0][j*5:j*5+2] - extract_val)*10
        if j < 10:
            future_data[i][0][j*5:j*5+2] = (future_data[i][0][j*5:j*5+2] - extract_val)*10

# show some data pairs after normalization and centralization

config = {
        "font.family":'Times New Roman',
        "font.size": 12,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
plt.rcParams.update(config)

plt.figure()
for i in range(50):
    # k = np.random.randint(0, history_data.shape[0])
    k = i
    for j in range(15):
        # show the legend of the history position
        if k == 19 and j == 0:
            # use royal blue to show the history position
            # plt.plot(history_data[k][0][j*5]/10, history_data[k][0][j*5+1]/10, 'ro', label='History Position', markersize=2)
            plt.plot(history_data[k][0][j*5]/10, history_data[k][0][j*5+1]/10, 'o', color='royalblue', label='History Position', markersize=2)
        else:
            # use royal blue to show the history position
            plt.plot(history_data[k][0][j*5]/10, history_data[k][0][j*5+1]/10, 'o', color='royalblue', markersize=2)
            # plt.plot(history_data[k][0][j*5]/10, history_data[k][0][j*5+1]/10, 'ro', markersize=2)
    for j in range(10):
        # show the legend of the future position
        if k == 19 and j == 0:
            # use seagreen to show the future position
            plt.plot(future_data[k][0][j*5]/10, future_data[k][0][j*5+1]/10, 'o', color='orangered', label='Future Position', markersize=2)
            # plt.plot(future_data[k][0][j*5]/10, future_data[k][0][j*5+1]/10, 'bo', label='Future Position', markersize=2)
        else:
            plt.plot(future_data[k][0][j*5]/10, future_data[k][0][j*5+1]/10, 'o', color='orangered', markersize=2)
            # plt.plot(future_data[k][0][j*5]/10, future_data[k][0][j*5+1]/10, 'bo', markersize=2)
plt.axis('equal')
# the x axis is the x axis in the map frame, the y axis is the y axis in the map frame
plt.xlabel('$x$ (m)')
plt.ylabel('$y$ (m)')
plt.legend()
plt.show()

# grep the first 3 elements (x,y,phi,vx,vy) -> (x,y,phi) of the future data to be the output
future_data = future_data.reshape((future_data.shape[0], 10, 5))
future_data = future_data[:, :, :3]
future_data = future_data.reshape((future_data.shape[0], 10*3))

X = np.concatenate(history_data, axis=0)
Y = future_data

# show some data pairs after normalization and centralization
# for i in range(5):
#     k = np.random.randint(0, X.shape[0])
#     for i in range(10):
#         plt.plot(Y[k][i*3], Y[k][i*3+1], 'bo')
#     for i in range(15):
#         plt.plot(X[k][i*5], X[k][i*5+1], 'ro')
#     plt.show()
# -------------------

# For demonstration purposes, I'll generate some random data
n_samples = X.shape[0]

if Train:
    # Define the kernel
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    # Create a Gaussian Process Regressor
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=10)
    # set the maximum number of iterations to 1000
    # Fit the Gaussian Process model
    tick = time.time()
    gpr.fit(X, Y)
    joblib.dump(gpr, 'gpr.pkl')
    tock = time.time()
    print('Time taken to fit the model: {} seconds'.format(tock - tick))
else:
    gpr = joblib.load('gpr.pkl')
    lstm = joblib.load('lstm.pkl')
    print('Model loaded')

# Now we can use the fitted model to make predictions
# For example, let's predict the future vehicle state vectors for a new history vehicle state vector
for i in range(5):
    plt.figure(figsize=(4, 3))
    k = np.random.randint(0, X.shape[0])
    new_history_vector = X[k].reshape(1, 15*5)
    new_pred_vector_gt = Y[k].reshape(1, 10*3)
    predicted_future_vector = gpr.predict(new_history_vector)
    # reshape the data from 75 to 15*5
    lstm_new_history_vector = new_history_vector.reshape((1, 15, 5))
    lstm_predicted_future_vector = lstm.predict(lstm_new_history_vector)
    print(predicted_future_vector)
    for i in range(10):
        plt.plot(predicted_future_vector[0][i*3]/10, predicted_future_vector[0][i*3+1]/10, 'bo')
        plt.plot(new_pred_vector_gt[0][i*3]/10, new_pred_vector_gt[0][i*3+1]/10, 'go')
        plt.plot(lstm_predicted_future_vector[0][i*3]/10, lstm_predicted_future_vector[0][i*3+1]/10, 'ro')
    for i in range(15):
        plt.plot(new_history_vector[0][i*5]/10, new_history_vector[0][i*5+1]/10, 'ko')
    plt.legend(['GPR prediction', 'Ground truth', 'LSTM prediction', 'History position'])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # equal axis
    plt.axis('equal')
    plt.show()