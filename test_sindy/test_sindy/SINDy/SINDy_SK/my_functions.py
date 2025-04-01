import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumtrapz
from scipy.integrate import cumtrapz

import os

from sklearn.metrics import mean_squared_error, explained_variance_score, max_error
from sklearn.preprocessing import StandardScaler
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import PolynomialLibrary


from pysindy.optimizers import SR3
from pysindy.optimizers import ConstrainedSR3
#import sdeint

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


def integrate_data(omega_data, dt=1, initial_theta=0):
    theta_data = cumtrapz(omega_data, dx=dt, initial=initial_theta)
    return theta_data

def split_into_chunks(data, region_name, chunk_size=3600):
    chunks = []
    num_chunks = len(data) // chunk_size
    
    for i in range(num_chunks):
        chunk = data.iloc[i*chunk_size:(i+1)*chunk_size].copy()
        chunk['Region'] = region_name
        chunks.append(chunk)
    
    return chunks


def calculate_model_scores(region_data, region_name, n=2):
    sindy_models = {}
    scores = {}

    transformations = {
        "exp_1000": lambda chunk: (np.exp((chunk.index - chunk.index[0]).total_seconds().to_numpy() / 1000) - 1).reshape(-1, 1),
        #"exp_3600": lambda chunk: (np.exp((chunk.index - chunk.index[0]).total_seconds().to_numpy() / 3600) - 1).reshape(-1, 1),

        # Sine transformation of time
        "sin_time": lambda chunk: np.sin((chunk.index - chunk.index[0]).total_seconds().to_numpy()).reshape(-1, 1),
        
        # Cosine transformation of time
        "cos_time": lambda chunk: np.cos((chunk.index - chunk.index[0]).total_seconds().to_numpy()).reshape(-1, 1),

        # Time itself
        "linear_time": lambda chunk: np.arange(0, len(chunk), 1).reshape(-1, 1),

        "cumsum_omega": lambda chunk: np.cumsum(np.abs(chunk['omega_filtered'].values)).reshape(-1, 1),
        "cumsum_theta": lambda chunk: np.cumsum(np.abs(integrate_data(chunk['omega_filtered'].values))).reshape(-1, 1),
        "diff_omega": lambda chunk: np.insert(np.diff(chunk['omega_filtered'].values), 0, 0).reshape(-1, 1),
        "diff_theta": lambda chunk: np.insert(np.diff(integrate_data(chunk['omega_filtered'].values)), 0, 0).reshape(-1, 1),

        
        "log_time": lambda chunk: np.log1p((chunk.index - chunk.index[0]).total_seconds().to_numpy()).reshape(-1, 1),
        "squared_time": lambda chunk: ((chunk.index - chunk.index[0]).total_seconds().to_numpy() ** 2).reshape(-1, 1),
    }

    for transform_name, transform_func in transformations.items():
        sindy_models[transform_name] = []
        scores[transform_name] = []

        for chunk in region_data:
            theta_chunk = integrate_data(chunk['omega_filtered'].values)
            stacked_data_chunk = np.column_stack((theta_chunk, chunk['omega_filtered'].values))
            
            # Apply the selected time transformation
            t_train_chunk = transform_func(chunk)

            x_train_augmented_chunk = np.hstack([stacked_data_chunk, t_train_chunk])
            
            feature_names_chunk = ["theta", "omega", "time"]
            polynomial_library_chunk = ps.PolynomialLibrary(degree=n)
            sparse_regression_optimizer_chunk = ps.STLSQ(threshold=1e-10)
            
            model_chunk = ps.SINDy(feature_names=feature_names_chunk, 
                                   feature_library=polynomial_library_chunk,
                                   optimizer=sparse_regression_optimizer_chunk)
            
            model_chunk.fit(x_train_augmented_chunk, t=1)
            sindy_models[transform_name].append(model_chunk)
            
            # Compute the explained variance score
            score = model_chunk.score(x_train_augmented_chunk, t=1, metric=explained_variance_score)
            scores[transform_name].append(score)

        # Compute the mean score for this transformation
        mean_score = np.mean(scores[transform_name])
        print(f"Mean Model Score for {region_name} using {transform_name}: {mean_score}")

    return sindy_models, scores

def simulate_sindy_model(model, initial_conditions, time_points, title, omega_original, omega_filtered):
    # Simulate the system using the provided model
    simulated_data = model.simulate(initial_conditions, time_points)

    # Extract simulated theta and omega
    simulated_theta = simulated_data[:, 0]
    simulated_omega = simulated_data[:, 1]

    # Plot the simulation results
    plt.figure(figsize=(10, 6))

    # Plot the original Omega with noise
    plt.plot(omega_original, label='Original Omega with noise', alpha=0.7, color='#2b6a99')

    # Plot the Filtered Omega
    plt.plot(omega_filtered, label='Filtered Omega', linestyle='-', linewidth=2, color='#1b7c3d')

    # Plot the simulated Omega
    plt.plot(simulated_omega, label='Simulated Omega', linestyle='--', linewidth=3, color='#f16c23')

    plt.title(f'Comparison of Original, Filtered, and Simulated Omega - {title}')
    plt.xlabel('Time')
    plt.ylabel('Omega')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()
