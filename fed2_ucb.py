import numpy as np
import matplotlib.pyplot as plt

## The communication phase is fixed to maintain a constant cost and ensure efficient data transfer.
def fp(p):  # Determines the length of the communication phase.
    fp = 100
    return int(fp)


def gp(p):  # Determines the length of the exploration phase.
    gp = 2 ** (p)
    return int(gp)


def spectrum_sensing(mu):
    idle_freq = []  # List to store indices of idle frequency bands

    for i in range(K):
        # Measure the energy level in the frequency band
        energy_level = measure_energy(mu, i)

        # If energy exceeds the idle threshold, consider it an idle band
        if energy_level > idle_threshold:
            idle_freq.append(i)

    return idle_freq


def measure_energy(mu, freq_index):
    # Measure the energy level in a specific frequency band using the global mean (mu).
    reward_level = mu[freq_index]  # Estimate reward level for the band
    return reward_level


N = 100  # Number of experiments
K = 10   # Number of frequency bands
global T
T = int(1e6)  # Total number of rounds
sigma = 1 / 2  # Noise level
sigma_c = 1 / 50  # Noise level for communication

comm_c = 0  # Communication cost accumulator
regret = np.zeros([N, T])
idle_threshold = 0.5  # Threshold to determine if a frequency is idle

# Randomly generate global mean rewards for each frequency band
mu_global = np.random.uniform(low=0, high=1, size=K)

# Perform spectrum sensing to identify idle frequencies
idle_freq = spectrum_sensing(mu_global)

# Start the repetition of experiments
for rep in range(N):
    t = 0  # Time step
    p = 1  # Phase step
    M = 0  # Number of local players

    # Set active arms to the idle frequencies identified
    active_arm = idle_freq
    C = 1  # Communication loss factor
    reward_global = np.zeros(T)  # Global reward
    optimal_reward = np.zeros(T)  # Optimal reward for comparison

    optimal_index = np.where(mu_global == np.max(mu_global))  # Identify optimal arm

    while t < T:
        """
        Start of round p
        """
        """
        Local players (devices)
        """
        if len(active_arm) > 1:
            player_add_num = gp(p)  # Determine number of players to add in this phase
            if M == 0:
                # If no players are initialized yet, create the first player
                M += 1
                pull_num = np.zeros([1, K])
                reward_local = np.zeros([1, K])
                mu_local = np.zeros([1, K])
                for k in range(K):
                    mu_local[M - 1, k] = np.random.normal(mu_global[k], sigma_c)  # Initialize local means
                player_add_num -= 1

            # Add new players based on gp(p)
            for m in range(player_add_num):
                M += 1
                pull_num = np.r_[pull_num, np.zeros([1, K])]
                reward_local = np.r_[reward_local, np.zeros([1, K])]
                mu_local = np.r_[mu_local, np.zeros([1, K])]
                for k in range(K):
                    mu_local[M - 1, k] = np.random.normal(mu_global[k], sigma_c)  # Generate local means for new players

        expl_len = fp(p)  # Length of the communication phase
        p += 1

        if len(active_arm) > 1:
            # Perform exploration and collect rewards
            for k in active_arm:
                for _ in range(min(T - t, expl_len)):
                    for m in range(M):
                        reward_local[m, k] += np.random.normal(mu_local[m, k], sigma)
                        pull_num[m, k] += 1
                    reward_global[t] = reward_global[t - 1] + M * np.random.normal(
                        mu_global[k], sigma_c
                    )
                    optimal_reward[t] = optimal_reward[t - 1] + M * np.random.normal(
                        mu_global[optimal_index][0], sigma_c
                    )
                    t += 1
            mu_local_sample = reward_local / pull_num  # Estimate local mean rewards

        # If only one arm is active, this means the optimal arm has been found
        if len(active_arm) == 1:
            # Exploit the best arm until the end of time
            reward_global[t:] = (
                reward_global[t - 1] + np.arange(T - t) * M * mu_global[active_arm[0]]
            )
            optimal_reward[t:] = (
                optimal_reward[t - 1] + np.arange(T - t) * M * mu_global[optimal_index]
            )
            break  # Exit loop when only one arm is active

        """
        Global server phase
        """
        if len(active_arm) > 1:
            reward_global[t - 1] -= M * C  # Apply communication cost (comment out to ignore this cost)
            comm_c += M  # Accumulate communication cost
            E = np.array([])  # Set of arms to be eliminated (based on UCB)

            # Estimate global mean reward from local samples
            mu_global_sample = 1 / M * sum(mu_local_sample)

            # Calculate exploration phase weight (eta_p) - differs from fed1
            eta_p = 0  
            for i in range(1, p):  # Sum over previous phases
                F_d = 0
                for j in range(i, p):
                    F_d += fp(j)
                eta_p += 1 / M**2 * gp(i) / F_d

            # Calculate confidence bound, tuned for better performance
            conf_bnd = np.sqrt(sigma**2 * eta_p * np.log(T)) + np.sqrt(
                sigma_c**2 * np.log(T) / M
            )

            # Identify the maximum element (arm) based on estimated rewards
            elm_max = np.nanmax(mu_global_sample) - conf_bnd
            for index in range(len(active_arm)):
                arm = active_arm[index]
                if mu_global_sample[arm] + conf_bnd < elm_max:
                    E = np.append(E, np.array([arm]))  # Add arm to elimination set

            # Eliminate arms with low rewards
            for i in range(len(E)):
                active_arm = np.delete(active_arm, np.where(active_arm == E[i]))

    regret[rep] = optimal_reward - reward_global  # Calculate regret for the current repetition

# Plot the results
plt.figure()
avg_regret = 1 / N * sum(regret)  # Average regret per experiment
err_regret = np.sqrt(np.var(abs(avg_regret - regret), axis=0))  # Error in regret calculation
print(
    "regret:",
    avg_regret[-1],  # Final regret at the last time step
    "comm:",
    comm_c / N,  # Average communication cost
    "delta:",
    round(np.sort(mu_global)[-1] - np.sort(mu_global)[-2], 3),  # Gap between the two best arms
)
plt.plot(range(T), avg_regret, label="Regret")
plt.xlabel("Number of Rounds")
plt.ylabel("Regret")
plt.legend(loc="best")
plt.show()
