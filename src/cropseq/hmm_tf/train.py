import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v2.enable_v2_behavior()

tfd = tfp.distributions

# https://www.tensorflow.org/probability/examples/Multiple_changepoint_detection_and_Bayesian_model_selection
true_rates = [40, 3, 20, 50]
true_durations = [10, 20, 5, 35]

observed_counts = tf.concat(
    [tfd.Poisson(rate).sample(num_steps)
     for (rate, num_steps) in zip(true_rates, true_durations)], axis=0)

# we already know the number of states based on previous experiment
num_states = 12
initial_state_logits = tf.zeros([num_states])  # uniform distribution

daily_change_prob = 0.05
transition_probs = tf.fill([num_states, num_states],
                           daily_change_prob / (num_states - 1))
transition_probs = tf.linalg.set_diag(transition_probs,
                                      tf.fill([num_states],
                                              1 - daily_change_prob))

print("Initial state logits:\n{}".format(initial_state_logits))
print("Transition matrix:\n{}".format(transition_probs))

# Define variable to represent the unknown log rates.
trainable_log_rates = tf.Variable(
    tf.math.log(tf.reduce_mean(observed_counts)) +
    tf.random.stateless_normal([num_states], seed=(42, 42)),
    name='log_rates')

hmm = tfd.HiddenMarkovModel(
    initial_distribution=tfd.Categorical(
        logits=initial_state_logits),
    transition_distribution=tfd.Categorical(probs=transition_probs),
    observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
    num_steps=len(observed_counts))

"""
Finally, we define the model's total log density, including a weakly-informative LogNormal prior on the rates, 
and run an optimizer to compute the maximum a posteriori (MAP) fit to the observed count data
"""
rate_prior = tfd.LogNormal(5, 5)

def log_prob():
 return (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) +
         hmm.log_prob(observed_counts))

losses = tfp.math.minimize(
    lambda: -log_prob(),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_steps=100)
