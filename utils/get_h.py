
def calculate_h(rewards, gamma =0.9):
    T_r = len(rewards)
    h_values = []
    for t in range(T_r):
        h_t = sum(gamma ** (k - t) * rewards[k] for k in range(t, T_r))
        h_values.append(h_t)
    return h_values