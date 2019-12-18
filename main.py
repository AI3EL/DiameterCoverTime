from agent import *
from env import *
import itertools
from tqdm import tqdm


def CT(l, d, options, nmax):
    agent = RWOptionAgent(options, l, d)
    env = Grid(d, l, [0]*d, [l-1]*d)
    times = {k: [] for k in itertools.combinations(np.ndindex(tuple([l]*d)), 2)}
    obs = env.reset()
    time = 0
    last_visit = -np.ones([l]*d, dtype=np.int)
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
                        times[(ts, tobs)].append(time-last_visit[ts])
                    else:
                        times[(tobs, ts)].append(time-last_visit[ts])
    print(last_visit)
    return times


d = 2
l = 5
options = [ActionOption(i, l, d) for i in range(2*d)]
times = CT(l, d, options, 10**5)

visited = {k: len(v) for k, v in times.items()}
avgs = {k: np.mean(v) for k, v in times.items()}

print(max(visited.values()))
print(min(visited.values()))
print(max(avgs.values()))
print(min(avgs.values()))
mx = max(avgs.values())
for k,v in avgs.items():
    if v == mx:
        print('Info on maximal time')
        print('Attained for state pair:', k)
        print('Number of times, pair was visited: ', visited[k])
        print('STD of times : ', np.std(times[k]))
        plt.hist(times[k], bins=50)
        plt.show()

f, axs = plt.subplots(1,2)
axs[0].hist(visited.values())
axs[1].hist(avgs.values())

plt.show()



