import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits
from scipy.special import wofz
from scipy.optimize import minimize

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)

def voigt(center,doppler,damp,ll):
    xx = (ll - center)/doppler
    return V(xx,1.0,damp)


# ME - underlying atmosphere:
def me(S1, S2, eta, vlos, deltav, loga, ll0, ll):
    
    center = ll0 * (1.0 + vlos / 3E5)
    doppler  = deltav/3E5 * ll0
    a = 10.0 ** loga
    profile = voigt(center, doppler, a, ll)
    
    return S1 + S2 / (1.0 + eta * profile)

# cloud that hangs above the atmosphere:
def cloud(S, deltatau, vlos, deltav, loga, ll0, ll, I_incoming):
    
    center = ll0 * (1.0 + vlos / 3E5) # line center in Angstroms, shifted due to velocity
    doppler  = deltav/3E5 * ll0 # Doppler width in angstroms
    
    a = 10.0 ** loga
    profile = voigt(center, doppler, a, ll)
    
    tau_lambda = deltatau * profile
    
    return I_incoming * np.exp(-tau_lambda) + S * (1.0 - np.exp(-tau_lambda))

# full model
def model_synth(p, ll0, ll):
    
    spectrum_atmos = me(p[0], p[1], p[2], p[3], p[4], p[5], ll0, ll)
    
    spectrum_final = cloud(p[6], p[7], p[8], p[9], p[10] ,ll0, ll, spectrum_atmos)
    
    return spectrum_final

#let's try now scipy.optimize.minimize

# just a normal chi2
def chi2(p, x, y, ll0, error):
    
    #x is ll
    #y are the observed stokes 
    #uncertanties in y
    
    y_model = model_synth(p,ll0, x)
    
    chi2 = np.sum(((y_model - y) / error)**2)
    
    return chi2

# chi2 trying to penalize the optical depth of the 

def chi2_r1(p, x, y, ll0, error):
    
    #x is ll
    #y are the observed stokes 
    #uncertanties in y
    
    y_model = model_synth(p,ll0, x)
    
    chi2 = np.sum(((y_model - y) / error)**2)
    
    return chi2

def invert_simple_py(ll, spectrum_to_fit, ll0, noise):
	
	# we are going to hard code the initial parameters and the bounds for the moment:
	
	# bounds:
	
	b=[(0.0,2.0), (-2.0,2.0),(80,120.), (-5,5),(1.0,20.0),(-4,1),(0, 2.0),(0,100),(-150, 150),(1.0, 15.0),(-4, -1)]
	params = np.array([0.10, 0.7, 100.0, 0.0, 2.0, -1, 0.1, 10.0, -30.0, 2.0, -4.0])
	result = minimize(chi2,params,args=(ll,spectrum_to_fit,ll0, noise), bounds=b)
	return result

def overseer_work(spectra_to_fit, ll, task_grain_size=16):
    """ Function to define the work to do by the overseer """

    # Reshape the atmosphere:
    NX,NY,NL = spectra_to_fit.shape
    spectra_to_fit = spectra_to_fit.reshape(NX*NY, NL)

    # Index of the task to keep track of each job
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    data_size = 0 # total number of pixels to invert
    num_tasks = 0 # ceiling of data_size / task_grain_size
    file_idx_for_task = [] # does this have sth to do with reading from file?
    task_start_idx = [] # no idea
    task_writeback_range = [] # no idea
    
    cdf_size = spectra_to_fit.shape[0]
    print("info::overseer::cdf_size = ", cdf_size)

    num_cdf_tasks = int(np.ceil(cdf_size / task_grain_size)) # number of tasks = roundedup number of pixels / grain
    
    task_start_idx.extend(range(0, cdf_size, task_grain_size))
    
    task_writeback_range.extend([slice(data_size + i*task_grain_size, min(data_size + (i+1)*task_grain_size,
        data_size + cdf_size)) for i in range(num_cdf_tasks)])
    
    data_size = cdf_size
    num_tasks = num_cdf_tasks

    # Define the lists that will store the data of each feature-label pair - I hate lists, can I work with 
    # numpy array 
    models = [None] * data_size
    fits = [None] * data_size
    
    success = True
    task_status = [0] * num_tasks

    with tqdm(total=num_tasks, ncols=110) as progress_bar:
        
        # While we don't have more closed workers than total workers keep looping
        while closed_workers < num_workers:
            data_in = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:
                try:
                    task_index = task_status.index(0)
                    
                    # Slice out our task
                    data = slice_tasks(spectra_to_fit, task_start_idx[task_index], task_grain_size)
                    data['index'] = task_index
                    data['ll'] = ll

                    # send the data of the task and put the status to 1 (done)
                    comm.send(data, dest=source, tag=tags.START)
                    task_status[task_index] = 1

                # If error, or no work left, kill the worker
                except:
                    comm.send(None, dest=source, tag=tags.EXIT)

            # If the tag is Done, receive the status, the index and all the data
            # and update the progress bar
            elif tag == tags.DONE:
                success = data_in['success']
                task_index = data_in['index']

                if not success:
                    task_status[task_index] = 0
                    print(f"Task: {task_index} failed")
                else:
                    task_writeback = task_writeback_range[task_index]
                    spectra[task_writeback] = data_in['spectrum']
                    progress_bar.update(1)

            # if the worker has the exit tag mark it as closed.
            elif tag == tags.EXIT:
                #print(" * Overseer : worker {0} exited.".format(source))
                closed_workers += 1

    # Once finished, dump all the data
    
    spectra = np.asarray(spectra)
    spectra = spectra.reshape(NX, NY, NL)
    print("info::overseer::writing the spectra")
    spechdu = fits.PrimaryHDU(spectra)
    wavhdu = fits.ImageHDU(wave)
    to_output = fits.HDUList([spechdu, wavhdu])
    to_output.writeto(sys.argv[1]+'_fit.fits', overwrite=True)

    models = np.asarray(models)
    models = models.reshape(NX, NY, NL)
    print("info::overseer::writing the spectra")
    mhdu = fits.PrimaryHDU(models)
    to_output.writeto(sys.argv[1]+'_fit.fits', overwrite=True)

