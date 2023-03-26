import librosa
import numpy as np
from cochleagram import cochleagram

def lif_coding(signal, time_constant, threshold):
    """Leaky integrate-and-fire coding of a signal.
        Parameters:
            signal (np.ndarray): The signal to be encoded.
            time_constant (float): The time constant of the leaky integrate-and-fire neuron.
            threshold (float): The threshold of the leaky integrate-and-fire neuron.
        Returns:
            np.ndarray: The spike train.
    """
    # Initialize the spike train.
    spike_train = np.zeros(len(signal))

    # Initialize the membrane potential.
    membrane_potential = np.zeros(len(signal)+1)

    # Initialize the membrane potential.
    membrane_potential[0] = signal[0,0]

    # Iterate through the time steps.
    for t in range(1, len(signal)):
        # Update the membrane potential.
        membrane_potential[t] = membrane_potential[t-1]*np.exp(-1/time_constant) + signal[:,t-1].sum()

        # Update the leaky integrate-and-fire neuron.
        # leaky_integrate_and_fire_neuron[0] = membrane_potential[t] / time_constant

        # If the leaky integrate-and-fire neuron spikes, generate a spike.
        if membrane_potential[t] >= threshold:
            spike_train[t-1] = 1
            membrane_potential[t] = 0
    return spike_train

# feat = cochleagram(
#     librosa.load("audio/fold1/203440-3-0-6.wav", sr=1000)[0],
#     n=8,
#     sr=1000,
#     low_lim=10,
#     hi_lim=20000,
#     sample_factor=2,
#     padding_size=None,
#     downsample=2,
#     nonlinearity=None,
#     fft_mode="auto",
#     ret_mode="envs",
#     strict=True,
# )
# print(lif_coding(feat, 0.001, 0.1))