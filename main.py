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
    axs[1].set_title('Histogram of the means E[T(s,s\')]')
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
    return env

# Cover-time computed by chapman-kolmogorov formula on markov chain induced by a random uniform policy on the mdp
def algebric_CT(mdp,states):
    P = torch.mean(mdp.P,dim=0) #transition matrix of a random uniform policy
    Ns = mdp.Ns
    T = torch.zeros(Ns,Ns)
    for s_prime in range(Ns):
        l=list(range(Ns))
        l.remove(s_prime)
        Q = P[l][:,l]
        vec = torch.mm(torch.inverse(torch.eye(Ns-1) - Q),torch.ones(Ns-1,1))
        vec = vec.reshape(-1).tolist()
        vec.insert(s_prime,0)
        T[s_prime] = torch.tensor(vec)
    print('Info on cover time')
    max = torch.max(T)
    max_pair = torch.where(T==max)
    max_pair = (max_pair[0].item(),max_pair[1].item())
    max_pair = (states[max_pair[0]],states[max_pair[1]])
    print('Attained for state pair:', max_pair)
    print('Value: ',max.item() )
    plt.hist(T[T>0].flatten(),bins=35)
    plt.title('Histogram of the means E[T(s,s\')]')
    plt.show()
    return T

def D_uniform_options(l, d, k, eps, nmax):
    agent = RWOptionAgent(options, l, d)
    env = Grid(d, l, [0] * d, [l - 1] * d, eps=eps)
    times = get_first_times(env, agent, nmax)



def random_transitions(Na,Ns,threshold):
    P = torch.zeros(Na,Ns,Ns)
    for a in range(Na):
        Pa = torch.rand(Ns,Ns)
        Pa[Pa < threshold] = 0
        Pa = Pa/torch.sum(Pa,axis=1)[:,None]
        P[a] = Pa
    return P

Na = 4
Ns = 10
threshold = 0.6
#P = random_tranisitions(Na,Ns,threshold)
#R = torch.zeros(Ns,Na)

d = 2
l = 5
options = [ActionOption(i, l, d) for i in range(2*d)]
grid = CT(l, d, options, 10**4)
mdp, states = Grid_to_MDP(grid)
algebric_CT(mdp,states)


