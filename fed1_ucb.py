import numpy as np
import matplotlib.pyplot as plt

def fp(p):
    """
    Returns exploration length for current round p.
    This can simulate the centralized setting by returning a constant value.
    """
    fp = 10 * np.log(T)
    # Uncomment the next line to simulate the centralized setting
    # fp = 1  
    return int(fp)

def spectrum_sensing(mu):
    """
    Perform spectrum sensing to detect idle frequency bands.
    Frequencies with an energy level above the idle threshold are considered idle.
    """
    idle_freq = []  # List to store indices of idle frequency bands

    for i in range(K):
        # Measure energy level of each frequency band
        energy_level = measure_energy(mu, i)

        # If energy level exceeds the threshold, mark the frequency as idle
        if energy_level > idle_threshold:
            idle_freq.append(i)

    return idle_freq

def measure_energy(mu, freq_index):
    """
    Measure the energy level of a specific frequency band.
    Returns a reward level based on the input global mean (mu).
    """
    reward_level = mu[freq_index]
    return reward_level

N = 100  # Number of repetitions
K = 10   # Number of frequency bands
C = 1    # Communication loss

global T
T = int(1e6)  # Total number of rounds
sigma = 1 / 2  # Standard deviation of noise
regret = np.zeros([N, T])  # Regret matrix initialized for N experiments

"""
# Uncomment this part to simulate the baseline algorithm with M=1 (centralized setting)
M = 1
all_matrix = np.ones([1,10])
"""

# Here we set M=5 to simulate the scenario where each player selects 2 arms (frequencies) 
# This setting prevents overlap between players, modeling a realistic cognitive radio (CR) scenario
M = 5
all_matrix = np.array([
    [4.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.2],
    [0.2, 4.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.2, 0.2],
    [0.2, 0.2, 4.2, 0.2, 0.2, 0.2, 0.2, 4.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 4.2, 0.2, 0.2, 4.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 4.2, 4.2, 0.2, 0.2, 0.2, 0.2],
])

"""
# Uncomment to simulate with M=10
M = 10
all_matrix = np.array([
    [9.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 9.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 9.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 9.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 9.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 9.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.1],
])
"""

"""
Originally, the global mean reward (mu_global) was hardcoded.
Here, we generate mu_global dynamically using random values to better model a realistic CR scenario.
"""
mu_global = np.random.uniform(low=0, high=1, size=K)
mu_local = mu_global * all_matrix  # Local rewards based on global mean
idle_threshold = 0.5  # Threshold to determine if a frequency is idle

# Call spectrum sensing to detect idle frequencies based on the global mean
idle_freq = spectrum_sensing(mu_global)

comm_c = 0  # Communication cost

# Simulation loop for N repetitions
for rep in range(N):
    t = 0
    p = 0
    active_arm = idle_freq  # Initialize active arms as idle frequencies
    pull_num = np.zeros([M, K])  # Track number of pulls for each arm
    reward_local = np.zeros([M, K])  # Local reward accumulation
    reward_global = np.zeros(T)  # Global reward
    optimal_reward = np.zeros(T)  # Optimal reward for comparison

    data_local = np.zeros([M, K, T])  # Simulated local data
    data_global = np.zeros([K, T])  # Simulated global data

    # Generate normal distributed rewards based on local means
    for j in range(M):
        for i in range(K):
            data_local[j, i] = np.random.normal(mu_local[j, i], sigma, T)

    # Identify the optimal arm
    optimal_index = np.where(mu_global == np.max(mu_global))
    for i in range(K):
        data_global[i] = np.random.normal(mu_global[i], sigma, T)

    # Main loop over time steps
    while t < T:
        # Exploration phase
        if len(active_arm) > 1:
            expl_len = fp(p)
            p += 1
            for k in active_arm:
                for _ in range(min(T - t, expl_len)):
                    for m in range(M):
                        reward_local[m, k] += data_local[m, k, t]
                        pull_num[m, k] += 1
                    reward_global[t] = reward_global[t - 1] + M * data_global[k, t]
                    optimal_reward[t] = (
                        optimal_reward[t - 1] + M * data_global[optimal_index, t]
                    )
                    t += 1
            mu_local_sample = reward_local / pull_num

        # Exploitation phase when only one arm is active
        if len(active_arm) == 1:
            reward_global[t:] = (
                reward_global[t - 1] + np.arange(T - t) * M * mu_global[active_arm[0]]
            )
            optimal_reward[t:] = (
                optimal_reward[t - 1] + np.arange(T - t) * M * mu_global[optimal_index]
            )
            break

        # Global server communication phase
        if len(active_arm) > 1:
            comm_c += M  # Accumulate communication loss
            reward_global[t - 1] -= (
                M * C
            )  # Comment out this line to ignore communication loss
            E = np.array([])  # Set of arms to be eliminated
            mu_global_sample = 1 / M * sum(mu_local_sample)
            conf_bnd = np.sqrt(
                4 * sigma**2 * np.log(T) / (M * pull_num[0, active_arm[0]])
            )  # Confidence bound adjusted for better performance
            elm_max = np.nanmax(mu_global_sample) - conf_bnd
            for index in range(len(active_arm)):
                arm = active_arm[index]
                if mu_global_sample[arm] + conf_bnd < elm_max:
                    E = np.append(E, np.array([arm]))

            # Eliminate arms from active_arm
            for i in range(len(E)):
                active_arm = np.delete(active_arm, np.where(active_arm == E[i]))

    # Calculate regret for the current repetition
    regret[rep] = optimal_reward - reward_global

# Compute average regret and error margin across repetitions
avg_regret = 1 / N * sum(regret)
err_regret = np.sqrt(np.var(abs(avg_regret - regret), axis=0))

# Output final statistics
print(
    "regret:",
    avg_regret[-1],  # Regret at the final time step
    "comm:",
    1 / N * comm_c,  # Average communication cost
    "delta:",
    round(np.sort(mu_global)[-1] - np.sort(mu_global)[-2], 3),  # Reward gap between best arms
)

# Plot the average regret over time
plt.figure()
plt.plot(range(T), avg_regret, label="Regret")
plt.xlabel("Number of Rounds")
plt.ylabel("Regret")
plt.legend(loc="best")
plt.show()
