import numpy as np
import matplotlib.pyplot as plt
import scienceplots
max_step = 32
lambda_ = 0.5

# Main curve with att_score = 0.5
att_score = 0
probabilities = []
for t in range(1, max_step + 1):
    S_t = lambda_ * np.sin(t * np.pi / max_step)
    prob = 1 - (t / max_step) - S_t * att_score
    probabilities.append(max(0, prob))  # Clip values to ensure they're not less than 0

# Range of att_score from 0 to 1
att_scores = np.linspace(0, 0.5, 100)
prob_ranges = []
for score in att_scores:
    prob_range = []
    for t in range(1, max_step + 1):
        S_t = lambda_ * np.sin(t * np.pi / max_step)
        prob = 1 - (t / max_step) - S_t * score
        prob_range.append(max(0, prob))  # Clip values to ensure they're not less than 0
    prob_ranges.append(prob_range)


with plt.style.context(['ieee']):
    plt.figure(figsize=(8,8),dpi=500)

    # Plotting
    plt.plot(range(1, max_step + 1), probabilities, label='att_score = 0.0, lambda = 0.5')

    att_score = 0.5
    probabilities = []
    for t in range(1, max_step + 1):
        S_t = lambda_ * np.sin(t * np.pi / max_step)
        prob = 1 - (t / max_step) - S_t * att_score
        probabilities.append(max(0, prob))  # Clip values to ensure they're not less than 0

    plt.plot(range(1, max_step + 1), probabilities, label='att_score = 0.5, lambda = 0.5')


    att_score = 0.25
    probabilities = []
    for t in range(1, max_step + 1):
        S_t = lambda_ * np.sin(t * np.pi / max_step)
        prob = 1 - (t / max_step) - S_t * att_score
        probabilities.append(max(0, prob))  # Clip values to ensure they're not less than 0

    plt.plot(range(1, max_step + 1), probabilities, label='att_score = 0.25, lambda = 0.5')

    plt.fill_between(range(1, max_step + 1), np.min(prob_ranges, axis=0), np.max(prob_ranges, axis=0), alpha=0.5,)
    plt.xlabel('Time Step')
    plt.ylabel('Probability')
    plt.title('Probability of Tokens Remaining Unmasked')
    plt.legend()
    plt.grid(True)
    plt.savefig('probOfTokenBeingMasked.pdf')
    plt.show()
