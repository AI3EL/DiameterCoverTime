import numpy as np


class Option:
    def __init__(self, init_states, pi, beta):
        self.pi = pi
        self.beta = beta
        self.init_states = init_states

    def is_in_init_states(self, s):
        return bool(self.init_states[s])


class ActionOption(Option):
    def __init__(self, action, l, d):
        state_shape = tuple([l]*d + [2*d])
        pi = np.zeros(state_shape)
        pi[..., action].fill(1)  # TODO: check if action or id of action
        beta = np.ones(tuple([l]*d))
        init_states = np.ones(tuple([l]*d))
        Option.__init__(self, init_states, pi, beta)


def list_actions(d):
    return list(range(-d,0)) + list(range(1,d+1))


def id_to_action(i, d):
    if i < d:
        return i-d
    else:
        return d-i+1


def action_to_id(a, d):
    if a < 0:
        return a+d
    else:
        return i+1-a


# option_pi.shape = (l^d, len(options))
class OptionAgent:
    def __init__(self, options, option_pi):
        self.options = options
        self.cur_option = None
        self.option_pi = option_pi
        self.check_consistency()

    # TODO: test
    def check_consistency(self):
        if len(self.options) != self.option_pi.shape[-1]:
            raise ValueError('Number of options : {}, options_pi.shape[-1] : {}'.
                             format(self.options, self.option_pi.shape[-1]))
        for s in np.ndindex(self.option_pi.shape[:-1]):
            pi_values = self.option_pi[s]
            for i,o in enumerate(self.options):
                if pi_values[i]>0 and not o.is_in_init_states(s):
                    raise ValueError('Option policy for state ', s,
                                     ' has positive value for an option that can\'t be initiated')

    def get_action(self, obs):
        d = len(obs)
        option_over = False
        if self.cur_option is None or np.random.rand() < self.cur_option.beta[tuple(obs)]:
            self.cur_option = np.random.choice(self.options, 1, replace=False, p=self.option_pi[tuple(obs)])[0]
            option_over = True
        probs = self.cur_option.pi[tuple(obs)]
        return np.random.choice(list_actions(d), 1, replace=False, p=probs/sum(probs))[0], option_over


class RWOptionAgent(OptionAgent):
    def __init__(self, options, l, d):
        option_pi = np.ones(tuple([l]*d + [len(options)]), dtype=np.float)
        option_pi /= len(options)
        OptionAgent.__init__(self, options, option_pi)