# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


## fp 일정한 이유: 전달 단계에서의 통신 비용을 일정하게 유지하고, 효율적인 데이터 전송을 보장하기 위함
def fp(p):  # communication phase 결정
    fp = 100
    return int(fp)


def gp(p):  # exploration phase 결정
    gp = 2 ** (p)
    return int(gp)


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
global T
T = int(1e6)
sigma = 1 / 2
sigma_c = 1 / 50

comm_c = 0
regret = np.zeros([N, T])
idle_threshold = 0.5  # 주파수 대역이 idle로 판단되는 임계값

# mu_global = np.array([0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.765, 0.77, 0.79])
mu_global = np.random.uniform(low=0, high=1, size=K)

# 주파수 대역이 idle한지 확인하기 위해 spectrum_sensing 함수 호출
idle_freq = spectrum_sensing(mu_global)

for rep in range(N):
    t = 0
    p = 1
    M = 0

    # active_arm = np.array(range(K), dtype=int)
    # idle한 주파수 대역을 active_arm으로 설정
    active_arm = idle_freq
    C = 1  # communication loss
    reward_global = np.zeros(T)
    optimal_reward = np.zeros(T)

    optimal_index = np.where(mu_global == np.max(mu_global))

    while t < T:
        """
        round p
        """
        """
        local players
        """
        if len(active_arm) > 1:
            player_add_num = gp(p)
            if M == 0:
                M += 1
                pull_num = np.zeros([1, K])
                reward_local = np.zeros([1, K])
                mu_local = np.zeros([1, K])
                for k in range(K):
                    mu_local[M - 1, k] = np.random.normal(
                        mu_global[k], sigma_c
                    )  # generated local mean
                player_add_num -= 1

            for m in range(player_add_num):
                M += 1
                pull_num = np.r_[pull_num, np.zeros([1, K])]
                reward_local = np.r_[reward_local, np.zeros([1, K])]
                mu_local = np.r_[mu_local, np.zeros([1, K])]
                for k in range(K):
                    mu_local[M - 1, k] = np.random.normal(
                        mu_global[k], sigma_c
                    )  # generated local mean

        expl_len = fp(p)
        p += 1

        if len(active_arm) > 1:
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
                    t = t + 1
            mu_local_sample = reward_local / pull_num

        # 모든 arm이 exploration phase를 마치고 최적 arm이 결정되었을 때를 의미
        if len(active_arm) == 1:
            reward_global[t:] = (
                reward_global[t - 1] + np.arange(T - t) * M * mu_global[active_arm[0]]
            )
            optimal_reward[t:] = (
                optimal_reward[t - 1] + np.arange(T - t) * M * mu_global[optimal_index]
            )
            break  # 모든 arm 결정, 고로 loop break
        """
        global server
        """
        if len(active_arm) > 1:
            reward_global[t - 1] -= (
                M * C
            )  # comment this line out to ignore communication loss
            comm_c += M
            E = np.array(
                []
            )  # E는 Exploration Set for arms that have lower reward than UCB
            mu_global_sample = 1 / M * sum(mu_local_sample)
            eta_p = 0  # fed1과 차이를 보이는 부분(exploration phase의 가중치) -> 이후 conf_bnd 계산에 포함
            for i in range(1, p):  # p has been added one above
                F_d = 0
                for j in range(i, p):
                    F_d += fp(j)
                eta_p += 1 / M**2 * gp(i) / F_d

            conf_bnd = np.sqrt(sigma**2 * eta_p * np.log(T)) + np.sqrt(
                sigma_c**2 * np.log(T) / (M)
            )  # the constants are tuned from the original ones in the paper to get better performance

            elm_max = np.nanmax(mu_global_sample) - conf_bnd
            for index in range(len(active_arm)):
                arm = active_arm[index]
                if mu_global_sample[arm] + conf_bnd < elm_max:
                    E = np.append(E, np.array([arm]))  # 이후에 exclude from active_arm

            for i in range(len(E)):
                active_arm = np.delete(active_arm, np.where(active_arm == E[i]))

    regret[rep] = optimal_reward - reward_global

plt.figure()
avg_regret = 1 / N * sum(regret)  # avg regret per experiment
err_regret = np.sqrt(np.var(abs(avg_regret - regret), axis=0))
print(
    "regret:",
    avg_regret[-1],
    "comm:",
    comm_c / N,
    "delta:",
    round(np.sort(mu_global)[-1] - np.sort(mu_global)[-2], 3),
)
plt.plot(range(T), avg_regret, label="Regret")
plt.xlabel("Number of Rounds")
plt.ylabel("Regret")
plt.legend(loc="best")
plt.show()
