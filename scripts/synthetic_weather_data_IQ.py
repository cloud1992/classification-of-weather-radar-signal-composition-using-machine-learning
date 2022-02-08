#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 13:29:35 2021

@author: Arturo Collado Rosell
email: arturo.collado.rosell@gmail.com
"""

import sys
import numpy as np
import numpy.matlib


def synthetic_IQ_data( **kwargs ):
    
    """ Function to create synthetic IQ weather data. The power spectral density (PSD) model for phenomenon and 
    clutter are gaussian functions. This function follows the ideas of the paper 
    "Simulation of weatherlike Doppler spectra and signals" by Dusan Zrnic, 1975
    
    Possible Inputs:
        
        clutter_power: power of clutter [times]
        clutter_s_w : clutter spectrum width [m/s] 
        phenom_power: power of phenom [times] 
        phenom_vm: phenom mean Doppler velocity [m/s]
        phenom_s_w: phenom spectrum width [m/s]
        noise_power: noise power [times] 
        M: number of samples in a CPI, the sequence length  
        wavelength: radar wavelength [m] 
        PRF: Pulse Repetition Frequecy [Hz]
        radar_mode: Can be "uniform" or "staggered" 
        int_stagg: list with staggered sequence, by default it is [2,3] when radar_mode == "staggered" 
        samples_factor: it is needed for windows effects, by default it is set to 10
        num_realizations: number of realization 
    
    Outputs:
        z_IQ: numpy array of shape [num_realization, M] with the complex IQ data
        time: time grid.
        
        possible calls:
            data, time = synthetic_IQ_data(**params), where params is a dictionary 
            with parameters name as keys
    
    """
    # some default parameters if not provided 
    if  'M' not in kwargs:
        M = 64     #default M value
    elif kwargs['M']%2 !=0:
        M = kwargs['M'] + 1 # if M is not an even number we add 1 to convert it to an even number 
    else:
        M = kwargs['M']        
    if 'radar_mode' not in kwargs:
        radar_mode = 'uniform'
    
    if 'radar_mode' in kwargs:
        radar_mode = kwargs['radar_mode']
        if radar_mode == 'staggered':
            if 'int_stagg' not in kwargs:
                int_stagg = [2, 3]
            else:
                int_stagg = kwargs['int_stagg']
    else:
        radar_mode = 'uniform'    
    
    if radar_mode == 'uniform':
        sample_index = np.arange(M)
        num_samples_uniform = M 
    else:
        num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) 
        sample_index = np.cumsum( np.matlib.repmat(int_stagg, 1, round(M/len(int_stagg))))
        sample_index = np.concatenate(([0], sample_index[:-1]))
        sample_index =  sample_index[:M]
    
    if 'samples_factor' not in kwargs:
        samples_factor = 10
        num_samples = samples_factor * num_samples_uniform
    
    if 'PRF' not in kwargs:
        raise Exception('Introduce a PRF value')
    else:
        PRF = kwargs['PRF']
    
    if 'wavelength' not in kwargs:
        wavelength = 3e8/5.6e9  
    else:
        wavelength = kwargs['wavelength']
    
          
    #data generation
    phenomFlag = False
    if 'phenom_power' in kwargs:
        phenomFlag = True
        phenom_power = kwargs['phenom_power']
        try:
            phenom_vm = kwargs['phenom_vm']
            phenom_s_w = kwargs['phenom_s_w']
            # converting the values to frequency space
            p_fm = - 2.0/wavelength * phenom_vm
            p_s_w_f = 2.0/wavelength * phenom_s_w
        except Exception as e:
            print(f"Exception: key{e} is not present as input")
            
        
    
    clutterFlag = False
    if 'clutter_power' in kwargs:
        clutterFlag = True
        clutter_power = kwargs['clutter_power']
        try:
            clutter_s_w = kwargs['clutter_s_w'] 
            # converting the values to frequency space
            c_s_w_f = 2.0/wavelength * clutter_s_w
        except Exception as e:
            print(f"Exception: key{e} is not present as input")
        
        
        
            
    noiseFlag = False
    if 'noise_power' in kwargs:
        noiseFlag = True
        noise_power = kwargs['noise_power']
    else:
        noise_power = 0

    if phenomFlag == False and clutterFlag == False and noiseFlag == False:
        print('There are not output data because you did not provide the necessary parameters inputs')

    #Frequency grid
    f = np.arange(-num_samples/2, num_samples/2) * PRF/num_samples      
    
    dep_p = np.zeros(num_samples,)
    if phenomFlag:
        idx_0 = int(np.round(p_fm * num_samples/PRF))
        freq_0 = p_fm - idx_0 * PRF/num_samples
        dep_p = phenom_power / np.sqrt(2 * np.pi)/ p_s_w_f * np.exp(-(f - freq_0)**2 / (2 * p_s_w_f**2))
        dep_p = np.roll(dep_p, idx_0)
        
    dep_c = np.zeros(num_samples,)
    if clutterFlag:
        
        dep_c = clutter_power / np.sqrt(2 * np.pi)/ c_s_w_f * np.exp(-(f )**2 / (2 * c_s_w_f**2))
          
        
    dep = dep_p + dep_c
    
    dep1 =  np.concatenate((dep[int(num_samples/2 + 1):], dep[0:int(num_samples/2 + 1)]))   
    
    lambdak = num_samples* PRF * dep1
    
    #Data generation using "Simulation of Weatherlike Doppler Spectra and Signals" paper
    if 'num_realizations' not in kwargs:
        num_realizations = 1
    else:
        num_realizations = kwargs['num_realizations']
    # Uniform distributed variable in (0, 1)
    U = np.random.rand(num_realizations,num_samples)
    # random variable with exponential distribution
    Pk = -1.0* np.matlib.repmat(lambdak, num_realizations, 1) * np.log(1 - U)
    
    #random phase to generate I and Q samples
    theta = 2 * np.pi * np.random.rand(num_realizations, num_samples)
    
    #Complex signal
    z_IQ =  np.fft.ifft(np.sqrt(Pk) * np.exp(1j * theta), axis = 1) 
    
    
    # truncating the output and adding the noise    
    z_IQ = z_IQ[:,sample_index] + np.sqrt(noise_power/2) * ( np.random.randn(num_realizations, M) + 1j* np.random.randn(num_realizations, M))   
    
    time = np.arange(num_samples)/PRF
    time = time[sample_index]
    
    return z_IQ, time


def PSD_estimation(z_IQ, PRF, w):
    """
    Inputs
    ----------
    z_IQ : numpy array
        complex IQ data.
    PRF : float
        Pulse Repetition Time [Hz].
    w : numpy array
        windows used.

    Returns
    -------
    psd : numpy array
        estimated DEP by means of periodograms.

    """
    U = sum(w**2) # window energy
    (I,M) = z_IQ.shape
    
    if I == 1:
        psd = 1/PRF/U * np.abs(np.fft.fft(z_IQ, axis = 1))**2
    else:
        psd = 1/PRF/U * np.sum(np.abs(np.fft.fft(z_IQ, axis = 1))**2, axis = 0 )
    return psd    

def PSD_estimation_parallel(z_IQ, PRF, w):
    """
    Inputs
    ----------
    z_IQ : numpy array
        complex IQ data.
    PRF : float
        Pulse Repetition Time [Hz].
    w : numpy array
        windows used.

    Returns
    -------
    psd : numpy array
        estimated DEP by means of periodograms.

    """
    U = sum(w**2) # window energy
    (I,M) = z_IQ.shape
    psd = 1/PRF/U * np.abs(np.fft.fft(z_IQ, axis = 1))**2
    
    # if I == 1:
    #     psd = 1/PRF/U * np.abs(np.fft.fft(z_IQ, axis = 1))**2
    # else:
    #     psd = 1/PRF/U * np.abs(np.fft.fft(z_IQ, axis = 1))**2
    return psd    
    
    
    
def zero_interpolation(z_IQ, int_stagg):
    
    """ 
    Input:
    ------
    z_IQ : numpy array
        complex IQ staggered data.
    int_stagg: array 
        staggered relationship e.g [2,3]    
        
    Return:
    ------
    z_IQ_interp: numpy array
        staggered data with zero added between samples    
    
    """
    (I,M) = z_IQ.shape
    num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) 
    sample_index = np.cumsum( np.matlib.repmat(int_stagg, 1, round(M/len(int_stagg))))
    sample_index = np.concatenate(([0], sample_index[:-1]))
    sample_index =  sample_index[:M]
    z_IQ_interp = np.zeros((I,num_samples_uniform), dtype = complex)
    z_IQ_interp[:,sample_index] = z_IQ
    return z_IQ_interp

def window_spectral_width(window, PRF):
    f_index = np.arange(-0.5, 0.5, 1/len(window))
    Sw = np.abs(np.fft.fftshift(np.fft.fft(window)))**2
    sw_window = np.sqrt(np.sum(f_index**2 * Sw)/ np.sum(Sw))
    return sw_window*PRF

def clutter_power(z_IQ, T1, T2, clutter_sw, window = 'Kaiser', alpha = 8):
    """ Function to estimate the clutter power 
        if T1 == T2 means uniform mode else staggered mode
        
        Inputs:
            z_IQ: complex IQ data
            T1: pulse repetition time (PRT) 1 [s]
            T2: PRT 2 [s]
            clutter_sw: asymptotic clutter spectral width [Hz]
            window: the windows, at the moment it could only be the Kaiser window
            alpha: the window parameter
        Output:
            clutter_power_e: estimated clutter power
    
    """
    I,M = z_IQ.shape
    if T1 == T2:
        Tu = T1
        num_samples_uniform = M
        sample_index = np.arange(0,num_samples_uniform,1)
        operation_mode = 'uniform'
    else:
        Tu = T2-T1
        n1, n2 = round(T1/Tu), round(T2/Tu)  
        int_stagg = [n1, n2]
        num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) 
        sample_index = np.cumsum( np.matlib.repmat(int_stagg, 1, round(M/len(int_stagg))))
        sample_index = np.concatenate(([0], sample_index[:-1]))
        sample_index =  sample_index[:M]
        operation_mode = 'staggered'
        
    if window == 'Kaiser':
        w = np.kaiser(num_samples_uniform, alpha)
    else:
        raise Exception('At the moment you can use a Kaiser window only')  
    # windowing the data    
    z_IQ_w = z_IQ * np.matlib.repmat( w[sample_index] , I, 1 )    
    data_zero_interp = np.zeros((I,sample_index[-1]+1), dtype = complex)
    data_zero_interp[:, sample_index] = z_IQ_w
    
    #PSD estimation
    psd = PSD_estimation_parallel(data_zero_interp, w = w[sample_index], PRF = 1/Tu)
    sw_window = window_spectral_width(w, PRF = 1/Tu)
    sw_clutter = np.sqrt(clutter_sw**2 + np.ones(clutter_sw.shape)*sw_window**2)
    df = 1/Tu/(M)
    if operation_mode == 'staggered':
        clutter_power_e = np.sqrt(2 * np.pi)* sw_clutter * (psd[:,0] + psd[:,-1] + psd[:,1]) / (1 + 2 * np.exp(- df**2 / (2 * clutter_sw**2)))
        clutter_power_e = clutter_power_e + 2* np.cos(2*np.pi/5)**2 + 2* np.cos(4*np.pi/5)**2 
    else:
        clutter_power_e = np.sqrt(2 * np.pi)* sw_clutter * (psd[:,0] + psd[:,-1] + psd[:,1]) / (1 + 2 * np.exp(- df**2 / (2 * clutter_sw**2)))

    return clutter_power_e
        
  
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\n" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
    
def synthetic_data_train(M, Fc, Tu = 0.25e-3, theta_3dB_acimut = 1, radar_mode = 'uniform', **kwargs):
    
    """ 
    Possible inputs:
    -------
    M: data length
    Fc: radar carrier frecuecy [Hz]
    Tu: pulse repetition time [s]. In the case of staggered [n1, n2], Tu = T2 - T1
    theta_3dB_acimut: acimut one-way half power width [degree]
    radar_mode: radar operation mode, it could be 'uniform' or 'staggered'
    int_stagg: staggered integers e.g [2, 3]
    N_vel: velocity classes
    N_spectral_w: spectral with classes
    N_csr: clutter to noise ratio classes. It account for the non-clutter class
    N_snr: signal to noise ratio classes.
    csr_interval: CSR interval for grid construcction [csr_min, csr_max] e.g [0, 50] [dB]
    s_w_interval: spectral width interval for grid construcction [s_w_min, s_w_max] * v_a, these two numbers must be fractionals one e.g [0.04, 0.4]*v_a, [m/s] 
    snr_interval: signal to noise ratio interval [snr_min, snr_max], e.g [0, 30] [dB]
    
    """
    np.random.seed(2021) # seed for reproducibility 
    wavelenght = 3e8/Fc
    v_a = wavelenght/(4.0*Tu)
    
    
    if 'N_snr' in kwargs:
        N_snr = kwargs['N_snr']
    else:
        N_snr = 15 
        
    if 'snr_interval' in kwargs:
        snr_interval_min = kwargs['snr_interval'][0]
        snr_interval_max = kwargs['snr_interval'][1]
    else:
        snr_interval_min = 0
        snr_interval_max = 30       
            
    if 'N_csr' in kwargs:
        N_csr = kwargs['N_csr']
    else:
        N_csr = 26        
        
    if 'csr_interval' in kwargs:
        csr_min = kwargs['csr_interval'][0]
        csr_max = kwargs['csr_interval'][1]
    else:
        csr_min = 0
        csr_max = 50    
    # specific parameters dependent of the operational mode
    if radar_mode == 'uniform':   
        num_samples_uniform = M
        # radar rotation speed    
        w =  theta_3dB_acimut * np.pi/180.0 / (M * Tu)
        if 'N_vel' in kwargs:
            N_vel = kwargs['N_vel']
        else:
            N_vel = 30
            
        if 'N_spectral_w' in kwargs:
            N_s_w = kwargs['N_spectral_w']
        else:
            N_s_w = 12
            
            
        if 's_w_interval' in kwargs:
            s_w_interval_min = kwargs['s_w_interval'][0]
            s_w_interval_max = kwargs['s_w_interval'][1]
        else:
            s_w_interval_min = 0.04
            s_w_interval_max = 0.4   
 
    elif radar_mode == 'staggered':
        if 'int_stagg' in kwargs:
            int_stagg = kwargs['int_stagg']
        else:
            int_stagg = [2, 3]

        # radar rotation speed
        w =  theta_3dB_acimut * np.pi/180.0 / (M * sum(int_stagg)/len(int_stagg)* Tu)
        if 'N_vel' in kwargs:
            N_vel = kwargs['N_vel']
        else:
            N_vel = 30
            
        if 'N_spectral_w' in kwargs:
            N_s_w = kwargs['N_spectral_w']
        else:
            N_s_w = 12
                             
        if 's_w_interval' in kwargs:
            s_w_interval_min = kwargs['s_w_interval'][0]
            s_w_interval_max = kwargs['s_w_interval'][1]
        else:
            s_w_interval_min = 0.04
            s_w_interval_max = 0.2          
            
        # staggered indexs
        num_samples_uniform = round((M - 1)* sum(int_stagg)/len(int_stagg)) 
        sample_index = np.cumsum( np.matlib.repmat(int_stagg, 1, round(M/len(int_stagg))))
        sample_index = np.concatenate(([0], sample_index[:-1]))
        sample_index =  sample_index[:M]
    else:
        raise Exception('The radar operation mode is not valid')

    
    clutter_spectral_width = w*wavelenght * np.sqrt(np.log(2.0)) / (2.0 * np.pi * theta_3dB_acimut* np.pi/180.0) # [m/s]
    

    
    ########################Clutter + noise ###################################
    # Clutter power grid
    Sc  = 1
    cnr_1 = np.linspace(-75, -20, 10)
    cnr_2 = np.linspace(0, 50, 10) #clutter to noise ratio
    cnr = np.concatenate((cnr_1, cnr_2))
    noise_power_grid = Sc /10.0**(cnr/10)
    L = 7200
    N_cnr = len(noise_power_grid)
    
    data_c_n = np.zeros(shape = (N_cnr * L, num_samples_uniform + 1), dtype = 'float32')
    
    #defining some imput parameters which are fixed
    parameters_c_n = {          'clutter_power':Sc,
                               'clutter_s_w': clutter_spectral_width,
                               'PRF':1/Tu,
                               'radar_mode':radar_mode,
                               'M':M,
                               'wavelenght':wavelenght,
                               'int_stagg':[2, 3],
                               'num_realizations':1
                               }   
    #windows
    window = np.kaiser(num_samples_uniform, 8)
    print('Synthetic data generation for training')
    # for loops
    for i in progressbar(range(N_cnr), 'Computing:'):
        
  
        ## update the input parameters
        parameters_c_n['noise_power'] = noise_power_grid[i]
                    
        for j in range(L):
                        
                        
                z_IQ, _ = synthetic_IQ_data(**parameters_c_n)
                        
                if radar_mode == 'staggered':
                    z_IQ_zero_interp = zero_interpolation(z_IQ, int_stagg)
                    data_w = z_IQ_zero_interp * window
                else:
                    data_w = z_IQ * window
                        
                        ##PSD estimation
                psd = PSD_estimation(data_w, w = window, PRF = 1/Tu)
                psd = psd /np.max(psd)
                data_c_n[i*L + j, :num_samples_uniform] = 10*np.log10(psd[0,:])
                
                if cnr[i] < -15:#noise only condition
                    data_c_n[i*L + j, num_samples_uniform] = 3 # the class 3 if for signal having only noise
                else:
                    data_c_n[i*L + j, num_samples_uniform] = 0 # the class 0 if for signal having clutter and noise
                
   

    #######################clutter + phenomenon + noise#########################
    Sp = 1
    # Clutter power grid
    csr = np.linspace(0, 50, 10)
    Sc_grid = Sp * 10**(csr/10)
    # snr grid
    snr = np.linspace(0,30,10)
    noise_pow_grid = Sp /10.0**(snr/10)
    # spectral width grid
    s_width_grid = np.linspace(s_w_interval_min, s_w_interval_max, N_s_w) * v_a
    # velocity grid
    vel_step = 2.0/N_vel * v_a
    vel_grid = -np.arange(-v_a + vel_step/2.0, v_a , vel_step)
    
    
    
    
    
    N_Sc = len(Sc_grid)
    N_s_w = len(s_width_grid)
    N_vel = len(vel_grid)
    N_snr = len(noise_pow_grid)
    L = 2
    
    
    data_c_p_n = np.zeros(shape = (N_Sc * N_s_w * N_vel * N_snr * L, num_samples_uniform + 1), dtype = 'float32')
    
    #defining some imput parameters which are fixed
    parameters_c_p_n = {'phenom_power':Sp,
                               'clutter_s_w': clutter_spectral_width,
                               'PRF':1/Tu,
                               'radar_mode':radar_mode,
                               'M':M,
                               'wavelenght':wavelenght,
                               'int_stagg':[2, 3],
                               'num_realizations':1
                               }   
    
    
    # looping through the four parameter grids
    for i in progressbar(range(N_vel), 'Computing:') :
       
        for s in range(N_Sc):
                            
            for n in range(N_snr):
                
                for q in range(N_s_w):
                    
                    ## update the input parameters
                    parameters_c_p_n['phenom_vm'] = vel_grid[i]
                    parameters_c_p_n['phenom_s_w'] = s_width_grid[q]
                    parameters_c_p_n['clutter_power'] = Sc_grid[s]
                    parameters_c_p_n['noise_power'] = noise_pow_grid[n]
                    
                    for j in range(L):
                        
                        
                        z_IQ, _ = synthetic_IQ_data(**parameters_c_p_n)
                        
                        if radar_mode == 'staggered':
                            z_IQ_zero_interp = zero_interpolation(z_IQ, int_stagg)
                            data_w = z_IQ_zero_interp * window
                        else:
                            data_w = z_IQ * window
                        
                        ##PSD estimation
                        psd = PSD_estimation(data_w, w = window, PRF = 1/Tu)
                        psd = psd /np.max(psd)
                        data_c_p_n[i*N_Sc*N_snr*N_s_w*L + s*N_snr*N_s_w*L + n*N_s_w*L + q*L + j, :num_samples_uniform] = 10*np.log10(psd[0,:])
                        data_c_p_n[i*N_Sc*N_snr*N_s_w*L + s*N_snr*N_s_w*L + n*N_s_w*L + q*L + j, num_samples_uniform] = 1
                                 
                        
    ######################Phenomenon + noise###################################
    L = 20
    data_p_n = np.zeros(shape = (N_s_w * N_vel * N_snr * L, num_samples_uniform + 1), dtype = 'float32')
    parameters_p_n = {         'phenom_power':Sp,                               
                               'PRF':1/Tu,
                               'radar_mode':radar_mode,
                               'M':M,
                               'wavelenght':wavelenght,
                               'int_stagg':[2, 3],
                               'num_realizations':1
                               }   
    
    
    for i in progressbar(range(N_vel), 'Computing:'):
        
        for n in range(N_snr):
                
            for q in range(N_s_w):
                    
                ## update the input parameters
                parameters_p_n['phenom_vm'] = vel_grid[i]
                parameters_p_n['phenom_s_w'] = s_width_grid[q]             
                parameters_p_n['noise_power'] = noise_pow_grid[n]
                    
                for j in range(L):
                        
                        
                    z_IQ, _ = synthetic_IQ_data(**parameters_p_n)
                        
                    if radar_mode == 'staggered':
                        z_IQ_zero_interp = zero_interpolation(z_IQ, int_stagg)
                        data_w = z_IQ_zero_interp * window
                    else:
                        data_w = z_IQ * window
                        
                    ##PSD estimation
                    psd = PSD_estimation(data_w, w = window, PRF = 1/Tu)
                    psd = psd /np.max(psd)
                    data_p_n[i*N_snr*N_s_w*L + n*N_s_w*L  + q*L + j, :num_samples_uniform] = 10*np.log10(psd[0,:])
                    data_p_n[i*N_snr*N_s_w*L + n*N_s_w*L  + q*L + j, num_samples_uniform] = 2
    
    
    
    data_PSD = np.concatenate((data_c_n, data_c_p_n , data_p_n ) , axis = 0)
    return data_PSD, radar_mode                                 
    print('Data generation has finished')                    
                        
   
    
    
    
    
    
    
    
    
    
    
   
