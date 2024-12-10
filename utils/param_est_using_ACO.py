import math
import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import the history data
data = np.load('data_w_real_steer_70_2.npy')
X = data[:, :-2]
U = data[:, -2:]

# Vehicle dynamics model
class VehicleDynamicsModel:
    def __init__(self, x, y, yaw, vx, vy, omega, acc, steer, kf, kr, lf, lr, m, Iz):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.acc = acc
        self.steer = steer
        self.kf = kf
        self.kr = kr
        self.lf = lf
        self.lr = lr
        self.m = m
        self.Iz = Iz
        self.Lk = lf * kf - lr * kr

    def update(self, dt):
        self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
        self.y += (self.vy * math.cos(self.yaw) + self.vx * math.sin(self.yaw)) * dt
        self.yaw += self.omega * dt
        self.vx += dt * self.acc
        self.vy = (self.m * self.vx * self.vy + dt * self.Lk * self.omega - dt * self.kf * self.steer *
                   self.vx - dt * self.m * (self.vx ** 2) * self.omega) / (self.m * self.vx - dt * (self.kf + self.kr))
        self.omega = (self.Iz * self.vx * self.omega + dt * self.Lk * self.vy - dt * self.lf *
                      self.kf * self.steer * self.vx) / (self.Iz * self.vx - dt * (self.lf ** 2 * self.kf + self.lr ** 2 * self.kr))

# Ant colony optimization
class AntColonyOptimizer:
    def __init__(self, n_ants, n_iterations, alpha, beta, evaporation_rate, q):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q

    def objective_function(self, params):
        kf, kr, lf, lr, m, Iz = params
        Lk = lf * kf - lr * kr
        cost = 0

        for i in range(len(X) - 1):
            x, y, yaw, vx, vy, omega = X[i]
            acc, steer = U[i]
            dt = 0.05

            model = VehicleDynamicsModel(x, y, yaw, vx, vy, omega, acc, steer, kf, kr, lf, lr, m, Iz)
            model.update(dt)

            x_pred, y_pred, yaw_pred, vx_pred, vy_pred, omega_pred = model.x, model.y, model.yaw, model.vx, model.vy, model.omega
            x_true, y_true, yaw_true, vx_true, vy_true, omega_true = X[i + 1]

            cost += ((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2 + (yaw_pred - yaw_true) ** 2 +
                     (vx_pred - vx_true) ** 2 + (vy_pred - vy_true) ** 2 + (omega_pred - omega_true) ** 2)

        return cost

    def optimize(self, bounds):
        best_params = None
        best_cost = float('inf')

        for _ in range(self.n_iterations):
            params_list = []
            costs_list = []

            for _ in range(self.n_ants):
                params = [random.uniform(low, high) for low, high in bounds]
                params[2] = 2.91 - params[3]  # Constraint: lf + lr = 2.91
                cost = self.objective_function(params)
                params_list.append(params)
                costs_list.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_params = params

            # Update pheromone levels
            for i in range(len(bounds)):
                low, high = bounds[i]
                delta_tau = self.q / best_cost
                bounds[i] = (low + delta_tau, high + delta_tau)

        return best_params, best_cost
# Initial values
# p0 = np.array([-120000, -120000, 1.40, 1.47, 1614, 2000])
# kf = -120000  # N/rad
# kr = -120000  # N/rad
# lf = 1.40  # m
# lr = 1.47  # m
# m = 1614  # kg
# Iz = 2000  # kg*m2s

p0 = np.array([-128916, -85944, 1.06, 1.81, 1614, 1536.7])
kf, kr, lf, lr, m, Iz = p0

# Parameter bounds
bounds = [(kf * 0.7, kf * 1.4), (kr * 0.7, kr * 1.4), (lf * 0.7, lf * 1.4),
          (lr * 0.7, lr * 1.4), (m * 0.7, m * 1.4), (Iz * 0.7, Iz * 1.4)]

# Ant colony optimization parameters
n_ants = 20
n_iterations = 100
alpha = 1
beta = 1
evaporation_rate = 0.1
q = 1

# Optimize
aco = AntColonyOptimizer(n_ants, n_iterations, alpha, beta, evaporation_rate, q)
best_params, best_cost = aco.optimize(bounds)

print(f"Best parameters: {best_params}")
print(f"Best cost: {best_cost}")

# Update model with optimized parameters
kf, kr, lf, lr, m, Iz = best_params








def vehicle_dynamics(x, u, p):
    dt = 0.05
    kf, kr, lf, lr, m, Iz = p
    Lk = lf * kf - lr * kr
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + (x[3] * math.cos(x[2]) - x[4] * math.sin(x[2])) * dt
    x_next[1] = x[1] + (x[4] * math.cos(x[2]) + x[3] * math.sin(x[2])) * dt
    x_next[2] = x[2] + x[5] * dt
    x_next[3] = x[3] + dt * u[0]
    x_next[4] = (m * x[3] * x[4] + dt * Lk * x[5] - dt * kf * u[1] * x[3] - dt * m * (x[3] ** 2) * x[5]) / (m * x[3] - dt * (kf + kr))
    x_next[5] = (Iz * x[3] * x[5] + dt * Lk * x[4] - dt * lf * kf * u[1] * x[3]) / (Iz * x[3] - dt * (lf ** 2 * kf + lr ** 2 * kr))
    return x_next

# Load data

data_test = np.load("data_w_real_steer_70_1.npy")
X_test = data_test[:, 0:6]
U_test = data_test[:, 6:8]
X_est_test = np.zeros_like(X_test)
X_est_test[0] = X_test[0]
for i in range(X_test.shape[0] - 1):
    X_est_test[i + 1] = vehicle_dynamics(X_est_test[i], U_test[i], best_params)
print(f'Average error: {np.mean(np.linalg.norm(X_test - X_est_test, axis=1))}')
print(f'Maximum error: {np.max(np.linalg.norm(X_test - X_est_test, axis=1))}')

fig = plt.figure()
sub1 = fig.add_subplot(221)
plt.plot(X_test[:,0]-X_est_test[:,0], label='Error of x')
plt.plot(X_test[:,1]-X_est_test[:,1], label='Error of y')
plt.legend()
sub2 = fig.add_subplot(222)
plt.plot(X_test[:,0], X_test[:,1], label='Ground truth of route')
plt.plot(X_est_test[:,0], X_est_test[:,1], label='Estimated of route')
plt.legend()
# test using pure estimated data
for i in range(X_test.shape[0] - 1):
    X_est_test[i + 1] = vehicle_dynamics(X_test[i], U_test[i], best_params)
print(f'Average error: {np.mean(np.linalg.norm(X_test - X_est_test, axis=1))}')
print(f'Maximum error: {np.max(np.linalg.norm(X_test - X_est_test, axis=1))}')
sub3 = fig.add_subplot(223)
plt.plot(X_test[:,0]-X_est_test[:,0], label='Error of x')
plt.plot(X_test[:,1]-X_est_test[:,1], label='Error of y')
plt.legend()
sub4 = fig.add_subplot(224)
plt.plot(X_test[:,0], X_test[:,1], label='Ground truth of route')
plt.plot(X_est_test[:,0], X_est_test[:,1], label='Estimated of route')
plt.legend()
plt.show()