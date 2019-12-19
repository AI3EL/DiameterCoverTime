from agent import *
from env import *
import itertools
from tqdm import tqdm


def get_first_times(env, agent, nmax):
    times = {k: [] for k in itertools.combinations(np.ndindex(tuple([l] * d)), 2)}
    obs = env.reset()
    time = 0
    last_visit = -np.ones([l] * d, dtype=np.int)
    last_visit[tuple(obs)] = time
    for _ in tqdm(range(nmax)):
        time += 1
        action, option_over = agent.get_action(obs)
        obs, done = env.step(action)
        if option_over:
            tobs = tuple(obs)
            last_visit[tobs] = time
            for s in np.ndindex(last_visit.shape):
                ts = tuple(s)
                if ts != tobs and last_visit[ts] > -1:
                    if (ts, tobs) in times.keys():
                        times[(ts, tobs)].append(time - last_visit[ts])
                    else:
                        times[(tobs, ts)].append(time - last_visit[ts])
    return times


def CT(l, d, options, nmax):
    agent = RWOptionAgent(options, l, d)
    env = Grid(d, l, [0]*d, [l-1]*d)
    times = get_first_times(env, agent, nmax)

    visited = {k: len(v) for k, v in times.items()}
    avgs = {k: np.mean(v) for k, v in times.items()}

    # Distribution of length
    f, axs = plt.subplots(1, 2)
    axs[0].hist(visited.values())
    axs[0].set_title('Distribution of the number of realizations of T(s,s\')')
    axs[1].hist(avgs.values())
    axs[1].set_title('Distribution of the averages of T(s,s\')')
    plt.show()

    mx = max(avgs.values())
    for k, v in avgs.items():
        if v == mx:
            print('Info on maximal time')
            print('Attained for state pair:', k)
            print('Number of times, pair was visited: ', visited[k])
            print('Average: ', mx)
            print('STD : ', np.std(times[k]))
            plt.hist(times[k], bins=50)
            plt.title('Distribution of CT')
            plt.show()


def D_uniform_options(l, d, k, eps, nmax):
    agent = RWOptionAgent(options, l, d)
    env = Grid(d, l, [0] * d, [l - 1] * d, eps=eps)
    times = get_first_times(env, agent, nmax)


d = 2
l = 5
options = [ActionOption(i, l, d) for i in range(2*d)]
CT(l, d, options, 10**4)




