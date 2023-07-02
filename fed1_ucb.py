# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def fp(p):
    fp = 10 * np.log(T)
    #    fp = 1 #simulate the centralized setting
    return int(fp)


def spectrum_sensing(mu):
    idle_freq = []  # idle한 주파수 대역의 인덱스를 저장하는 리스트

    for i in range(K):
        # 주파수 대역에서의 에너지 수준 측정
        energy_level = measure_energy(mu, i)

        if energy_level > idle_threshold:
            idle_freq.append(i)

    return idle_freq


def measure_energy(mu, freq_index):
    # 특정 주파수 대역에서의 에너지 수준을 측정하는 함수
    # mu 값을 활용하여 에너지 수준을 추정

    # mu 값을 활용하여 주파수 대역의 보상 값을 추정
    reward_level = mu[freq_index]

    return reward_level


N = 100  # repeat times
K = 10
C = 1  # communication loss

global T
T = int(1e6)
sigma = 1 / 2
regret = np.zeros([N, T])

"""
M=1 #uncomment this part to run the baseline algorithm
all_matrix = np.ones([1,10])
"""

# 각 플레이어마다 2개의 arm(주파수)을 선택하는 방식으로 가중치를 부여한 경우
# 겹치지 않도록 하여 CR 시나리오에 부합한다고 볼 수 있음
M = 5  # uncomment this part to run with M=5
all_matrix = np.array(
    [
        [4.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.2],
        [0.2, 4.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 4.2, 0.2],
        [0.2, 0.2, 4.2, 0.2, 0.2, 0.2, 0.2, 4.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 4.2, 0.2, 0.2, 4.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 4.2, 4.2, 0.2, 0.2, 0.2, 0.2],
    ]
)

"""
M =10 #uncomment this part to run with M=10
all_matrix = np.array([[9.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                    [0.1,9.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                    [0.1,0.1,9.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,9.1,0.1,0.1,0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1,9.1,0.1,0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1,0.1,9.1,0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1,0.1,0.1,9.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1,0.1,0.1,0.1,9.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,9.1,0.1],
                    [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,9.1]
                    ])
"""

"""
원래는 하드코딩 되어있던 부분이므로 랜덤으로 mu_global을 생성하도록 수정
따라서 mu_global의 값이 동적으로 계속 바뀌어 CR 시나리오에 좀 더 부합
"""
# global mean reward
# mu_global = np.array([0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.765, 0.77, 0.79])
mu_global = np.random.uniform(low=0, high=1, size=K)
mu_local = mu_global * all_matrix
idle_threshold = 0.5  # 주파수 대역이 idle로 판단되는 임계값

# 주파수 대역이 idle한지 확인하기 위해 spectrum_sensing 함수 호출
idle_freq = spectrum_sensing(mu_global)


comm_c = 0

for rep in range(N):
    t = 0
    p = 0

    # active_arm = np.array(range(K), dtype=int)
    # idle한 주파수 대역을 active_arm으로 설정
    active_arm = idle_freq
    pull_num = np.zeros([M, K])
    reward_local = np.zeros([M, K])
    reward_global = np.zeros(T)
    optimal_reward = np.zeros(T)

    data_local = np.zeros([M, K, T])  # M*K*T
    data_global = np.zeros([K, T])  # K*T

    for j in range(M):
        for i in range(K):
            data_local[j, i] = np.random.normal(mu_local[j, i], sigma, T)

    optimal_index = np.where(mu_global == np.max(mu_global))
    for i in range(K):
        data_global[i] = np.random.normal(mu_global[i], sigma, T)

    while t < T:
        """
        round p
        """

        """
        local players
        """
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
                    t = t + 1
            mu_local_sample = reward_local / pull_num

        if len(active_arm) == 1:
            reward_global[t:] = (
                reward_global[t - 1] + np.arange(T - t) * M * mu_global[active_arm[0]]
            )
            optimal_reward[t:] = (
                optimal_reward[t - 1] + np.arange(T - t) * M * mu_global[optimal_index]
            )
            break

        """
        global server
        """
        if len(active_arm) > 1:
            comm_c += M  # 누적 통신 손실
            reward_global[t - 1] -= (
                M * C
            )  # comment this line out to ignore communication loss
            E = np.array([])
            mu_global_sample = 1 / M * sum(mu_local_sample)
            conf_bnd = np.sqrt(
                4 * sigma**2 * np.log(T) / (M * pull_num[0, active_arm[0]])
            )  # the constants are tuned from the original ones in the paper to get better performance
            elm_max = np.nanmax(mu_global_sample) - conf_bnd
            for index in range(len(active_arm)):
                arm = active_arm[index]
                if mu_global_sample[arm] + conf_bnd < elm_max:
                    E = np.append(E, np.array([arm]))

            for i in range(len(E)):
                active_arm = np.delete(active_arm, np.where(active_arm == E[i]))

    regret[rep] = optimal_reward - reward_global

avg_regret = 1 / N * sum(regret)  # avg regret per experiment
err_regret = np.sqrt(np.var(abs(avg_regret - regret), axis=0))
print(
    "regret:",
    avg_regret[-1],  # last time's regret(실험이 종료된 후의 regret)
    "comm:",
    1 / N * comm_c,
    "delta:",
    round(np.sort(mu_global)[-1] - np.sort(mu_global)[-2], 3),
)
plt.figure()
plt.plot(range(T), avg_regret, label="Regret")
plt.xlabel("Number of Rounds")
plt.ylabel("Regret")
plt.legend(loc="best")
plt.show()
