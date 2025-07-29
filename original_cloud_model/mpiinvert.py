import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits
from scipy.special import wofz
from scipy.optimize import minimize

import cloud_model as cm
from tqdm import tqdm

from threadpoolctl import threadpool_limits

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
threadpool_limits(1)
import pickle
from enum import IntEnum
from mpi4py import MPI
import sys
threadpool_limits(1)

class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """ # Makes sense to me, but not sure what is the IntEnum class 
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def slice_tasks(datain, task_start, grain_size):
    
    task_end = min(task_start + grain_size, datain.shape[0])

    sl = slice(task_start, task_end) # this is a slice object, allowing us to access the specific thingy
    
    data = {}
    data['taskGrainSize'] = task_end - task_start

    data['spectra'] = datain[sl,:]

    return data

def overseer_work(specarray, wave, task_grain_size=16):
    """ Function to define the work to do by the overseer """

    # Reshape the spectra:
    NX,NY,NL = specarray.shape
    specarray = specarray.reshape(NX*NY, NL)

    # Index of the task to keep track of each job
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    data_size = 0 # Let's figure out what this is - total number of pixels?
    num_tasks = 0 # And this is data_size // 16? 
    file_idx_for_task = [] # does this have sth to do with reading from file?
    task_start_idx = [] # no idea
    task_writeback_range = [] # no idea
    
    cdf_size = NX * NY
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
    spectral_fits = [None] * data_size
    
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
                    data = slice_tasks(specarray, task_start_idx[task_index], task_grain_size)
                    data['index'] = task_index
                    data['wave'] = wave
                    
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
                    models[task_writeback] = data_in['models']
                    spectral_fits[task_writeback] = data_in['fits']
                    progress_bar.update(1)

            # if the worker has the exit tag mark it as closed.
            elif tag == tags.EXIT:
                #print(" * Overseer : worker {0} exited.".format(source))
                closed_workers += 1

    models = np.asarray(models)
    spectral_fits = np.asarray(spectral_fits)
    # Once finished, dump all the data
    modhdu = fits.PrimaryHDU(models.reshape(NX, NY, -1))
    fithdu = fits.ImageHDU(spectral_fits.reshape(NX, NY, -1))
    wavhdu = fits.ImageHDU(wave)
    to_output = fits.HDUList([modhdu, fithdu, wavhdu])
    to_output.writeto(sys.argv[1]+'_inverted.fits', overwrite=True)

def worker_work(rank):
    # Function to define the work that the workers will do

    while True:
        # Send the overseer the signal that the worker is ready
        comm.send(None, dest=0, tag=tags.READY)
        # Receive the data with the index of the task, the atmosphere parameters and/or the tag
        data_in = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            
            # Receive the spectrum to invert:
            task_index = data_in['index'] # I think we need this? - for what though (to keep track of what succeeeded where)
            wave = data_in['wave'].astype(float)
            spectra_to_fit = data_in['spectra']
            task_size = data_in['taskGrainSize']
            ll0 = 6562.8
            
            models = np.zeros([task_size, 11])
            spectral_fits = np.zeros([task_size, len(wave)])
            
            for t in range(task_size): # Now, task size is not one, probably a good choice
                # Configure the context
                
                success = 1
                #try:
                    # This should work
                result = cm.invert_simple_py(wave, spectra_to_fit[t], ll0, 1E-4)
                
                #except:
                    # NOTE(cmo): In this instance, the task should never fail
                    # for sane input.
                #    success = 0
                #    break
                models[t,:] = np.copy(result.x)
                spectral_fits[t,:] = np.copy(cm.model_synth(models[t,:], 6562.8, wave))


            # Send the computed data
            # we do want to fill in tau too, but that can wait for the next step
            data_out =  {'index': task_index, 'success': success, 'models' : models, 'fits' : spectral_fits}
            comm.send(data_out, dest=0, tag=tags.DONE)

        # If the tag is exit break the loop and kill the worker and send the EXIT tag to overseer
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)

# simple inversion here. to be replaced with the mpi thing:

# load the data:

if (__name__ == '__main__'):

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    if rank == 0:

        # --------------------------------------------------------------------
        specarray = 0
        print("info::overseer::loading the data...: ")
        specarray = fits.open("/home/milic/data/scratch/mihi_data.fits")[0].data[15:-15,15:-15,0,45:525]
        ll = fits.open("/home/milic/data/scratch/mihi_data.fits")[1].data[45:525]

        
        print("info::overseer:: spectra shape is: ", specarray.shape)

        overseer_work(specarray, ll, task_grain_size = 16)
    else:
        worker_work(rank)
        pass

    # This here is the single process stuff 
'''
data = fits.open("/home/milic/data/scratch/mihi_data.fits")[0].data
print ("info::shape of the observed data cube is: ", data.shape)

plt.clf()
plt.figure(figsize=[5,4])
plt.imshow(np.sum(data[:,:,0,200:205], axis =2).T, origin='lower', cmap='grey')
plt.savefig("test_field.png", bbox_inches='tight')

lmin = 45
lmax = 525
ll = fits.open("/home/milic/data/scratch/mihi_data.fits")[1].data[lmin:lmax]
ll0 = 6562.8

from scipy.optimize import minimize
i = 44
j = 54

spectrum_to_fit = data[i,j,0,lmin:lmax]
result = cm.invert_simple_py(ll, spectrum_to_fit, ll0, 1E-4)

print (result.x)

plt.clf()
plt.figure(figsize=[8,5])
plt.plot(spectrum_to_fit)
plt.plot(cm.model_synth(result.x, ll0, ll))
plt.savefig("test_fit.png", bbox_inches='tight')
'''



