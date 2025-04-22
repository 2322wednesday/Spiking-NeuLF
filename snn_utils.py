import numpy as np
import torch
from tqdm import tqdm
import random
from datetime import datetime

def genUniformIntervalSpikeTrain(n_neurons: int, rates: list, dt: float, period: float, offset: float):
    """
    Generates spike trains with uniform intervals based on input parameters.
    
    Parameters:
    - n_neurons: Number of neurons
    - rates: List of firing rates (Hz) for each neuron
    - dt: Time resolution (seconds)
    - period: Duration of the spike train (seconds)
    - offset: Time offset for spike times (seconds)
    
    Returns:
    - spike_time: Array of spike times
    - spike_idx: Array of corresponding neuron indices
    """
    neurons = np.arange(n_neurons)

    spike_time = []
    spike_idx = []

    if len(neurons) == len(rates):
        for idx, rate in enumerate(rates):
            if rate > 0:
                interval = 1 / rate  # Time interval between spikes (seconds)
                n_spikes = int(period / interval)  # Number of spikes
                spike_time_temp = np.arange(0, n_spikes * interval, interval)[:n_spikes] + offset
                spike_idx_temp = np.full(len(spike_time_temp), neurons[idx])

                spike_time.extend(spike_time_temp)
                spike_idx.extend(spike_idx_temp)
    else:
        raise Exception('ERROR: The number of neurons does not match the number of rates.')

    # Convert to numpy arrays
    spike_time = np.array(spike_time)
    spike_idx = np.array(spike_idx)

    return spike_time, spike_idx

def genPoissonSpikeTrain(n_neurons: int, rates: list, dt: float, period: float, offset: float):

    #random.seed(datetime.now())
    random.seed(1004)

    n_samples = int(period/dt)

    neurons = np.linspace(0, n_neurons-1, n_neurons)

    spike_time = np.zeros(n_samples * len(neurons))
    spike_idx = np.zeros(n_samples * len(neurons))

    write_idx = 0

    if len(neurons) == len(rates):
        for idx in range(len(neurons)):
            spike_time_temp = np.where(np.random.random_sample(n_samples) < rates[idx] * dt)[0] * dt + offset
            spike_idx_temp = np.ones(len(spike_time_temp)) * neurons[idx]

            spike_time[write_idx : write_idx + len(spike_time_temp)] = spike_time_temp
            spike_idx[write_idx : write_idx + len(spike_idx_temp)] = spike_idx_temp

            write_idx += len(spike_time_temp)
    
    else:
        raise Exception('ERROR : The number of neurons does not match the number of rates.')

    spike_time = spike_time[:write_idx]
    spike_idx = spike_idx[:write_idx]

    return spike_time, spike_idx

def neuron_input_layer(self, input_shape):
        """
        Generate input layer neurons.

        Args:
            input_shape (_type_): _description_
        """
        self.neurons['input_layer'] = np.prod(input_shape[1:])

def encodeSpike(self, data, label, spike_duration, time_interval, mode):
        if mode == 'poisson':
            pass
        elif mode == 'uniform':
            pass
        else:
            raise Exception(f"Warnng: The spike encoding mode has been entered incorrectly: {mode}")

        self.spike_duration = spike_duration
        self.time_interval = time_interval
        ref = 0

        spike_time = np.zeros(len(data) * self.neurons['input_layer'] * np.int64(self.spike_duration/(0.001)))
        spike_idx = np.zeros(len(data) * self.neurons['input_layer'] * np.int64(self.spike_duration/(0.001)))
        
        for i in tqdm(range(len(data))):
            
            if isinstance(data[i], torch.Tensor):
                rates = data[i].numpy().flatten()
            elif isinstance(data[i], np.ndarray):
                rates = data[i].flatten()
            else:
                raise TypeError(f"Unsupported data type: {type(data[i])}")

            if mode == 'uniform':
                time, idx = genUniformIntervalSpikeTrain(n_neurons=self.neurons['input_layer'], 
                                                    rates=rates, 
                                                    dt=0.001, 
                                                    period=self.spike_duration, 
                                                    offset=i*self.spike_duration)
            elif mode == 'poisson':
                time, idx = genPoissonSpikeTrain(n_neurons=self.neurons['input_layer'], 
                                                   rates=rates, 
                                                   dt=0.001, 
                                                   period=self.spike_duration, 
                                                   offset=i*self.spike_duration)

            time += self.time_interval*i
            spike_time[ref : ref + len(time)] = time
            spike_idx[ref : ref + len(idx)] = idx
            
            ref += len(time)

        self.spike_time = spike_time[:ref]
        self.spike_idx = spike_idx[:ref]

        self.label = label
        self.data_num = len(data)
        self.total_duration = (self.spike_duration+self.time_interval)*self.data_num*b2.second