import numpy as np
import torch


# ====================================================================
def fvoigt(damp, vv):
    """
    Voigt function approximation using torch tensors.
    Based on: https://github.com/aasensio/neural_fields/blob/main/utils.py#L174
    """
    A = [122.607931777104326, 214.382388694706425, 181.928533092181549,
         93.155580458138441, 30.180142196210589, 5.912626209773153,
         0.564189583562615]

    B = [122.60793177387535, 352.730625110963558, 457.334478783897737,
         348.703917719495792, 170.354001821091472, 53.992906912940207,
         10.479857114260399, 1.]

    z = damp - torch.abs(vv) * 1j

    Z = ((((((A[6] * z + A[5]) * z + A[4]) * z + A[3]) * z + A[2]) * z + A[1]) * z + A[0]) / \
        (((((((z + B[6]) * z + B[5]) * z + B[4]) * z + B[3]) * z + B[2]) * z + B[1]) * z + B[0])

    h = Z.real
    f = torch.sign(vv) * Z.imag * 0.5

    return [h, f]


# ====================================================================
def me(S1, S2, eta, vlos, deltav, loga, ll0, ll):
    """
    ME - underlying atmosphere model.
    Vectorized implementation.
    
    Args:
        S1, S2, eta, vlos, deltav, loga, ll0: atmosphere parameters
        ll: wavelength array
    
    Returns:
        Synthetic spectrum from the underlying atmosphere
    """
    center = ll0 * (1.0 + vlos / 3E5)
    doppler = torch.abs(deltav) / 3E5 * ll0
    a = 10.0 ** loga
    xx = (ll[None, :] - center[:, None]) / doppler[:, None]
    H, F = fvoigt(a[:, None], xx)
    profile = H
    
    return torch.abs(S1[:, None]) + torch.abs(S2[:, None]) / (1.0 + torch.abs(eta[:, None]) * profile * 1000.0)


# ====================================================================
def cloud(S, deltatau, vlos, deltav, loga, ll0, ll, I_incoming):
    """
    Cloud model that hangs above the atmosphere.
    
    Args:
        S: source function
        deltatau: optical depth
        vlos: line-of-sight velocity
        deltav: velocity dispersion
        loga: damping parameter (log scale)
        ll0: line center wavelength
        ll: wavelength array
        I_incoming: incoming intensity
    
    Returns:
        Synthetic spectrum with cloud absorption
    """
    center = ll0 * (1.0 + vlos / 3E5)  # line center in Angstroms, shifted due to velocity
    doppler = torch.abs(deltav) / 3E5 * ll0  # Doppler width in angstroms
    
    a = 10.0 ** loga
    xx = (ll[None, :] - center[:, None]) / doppler[:, None]
    H, F = fvoigt(a[:, None], xx)
    profile = H
    
    tau_lambda = torch.abs(deltatau[:, None]) * profile
    
    return I_incoming * torch.exp(-tau_lambda) + torch.abs(S[:, None]) * (1.0 - torch.exp(-tau_lambda))


# ====================================================================
def model_synth(x, p):
    """
    Full model synthesis with one cloud layer.
    
    Args:
        x: [wavelength_array, line_center_array]
        p: parameter array with shape (n_pixels, 11)
           [S1, S2, eta, vlos_atmos, deltav_atmos, loga_atmos, 
            S_cloud, deltatau, vlos_cloud, deltav_cloud, loga_cloud]
    
    Returns:
        Synthetic spectrum
    """
    ll, ll0 = x
    
    ll = torch.from_numpy(ll.astype(np.float32))
    ll0 = torch.from_numpy(ll0.astype(np.float32))
    
    # Same device:
    if ll.device != p.device:
        ll = ll.to(p.device)
    if ll0.device != p.device:
        ll0 = ll0.to(p.device)
    

    # Atmosphere spectrum
    spectrum_atmos = me(p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 5], ll0, ll)
    
    # Cloud spectrum
    spectrum_final = cloud(p[:, 6], p[:, 7], p[:, 8], p[:, 9], p[:, 10], ll0, ll, spectrum_atmos)
    
    return spectrum_final


# ====================================================================
def model_synth_2clouds(x, p):
    """
    Full model synthesis with two cloud layers.
    
    Args:
        x: [wavelength_array, line_center_array]
        p: parameter array with shape (n_pixels, 16)
           [S1, S2, eta, vlos_atmos, deltav_atmos, loga_atmos,
            S_cloud1, deltatau1, vlos_cloud1, deltav_cloud1, loga_cloud1,
            S_cloud2, deltatau2, vlos_cloud2, deltav_cloud2, loga_cloud2]
    
    Returns:
        Synthetic spectrum
    """
    ll, ll0 = x
    
    ll = torch.from_numpy(ll.astype(np.float32))
    ll0 = torch.from_numpy(ll0.astype(np.float32))
    
    # Same device:
    if ll.device != p.device:
        ll = ll.to(p.device)
    if ll0.device != p.device:
        ll0 = ll0.to(p.device)
    
    # Atmosphere spectrum
    spectrum_atmos = me(p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 5], ll0, ll)
    
    # First cloud layer
    spectrum_cloud1 = cloud(p[:, 6], p[:, 7], p[:, 8], p[:, 9], p[:, 10], ll0, ll, spectrum_atmos)
    
    # Second cloud layer
    spectrum_final = cloud(p[:, 11], p[:, 12], p[:, 13], p[:, 14], p[:, 15], ll0, ll, spectrum_cloud1)
    
    return spectrum_final

def model_synth_2clouds_givenbck(x, p, I_incoming):

    ''' Full model synthesis with two cloud layers, given background intensity.

    Args:
        x: [wavelength_array, line_center_array]
        p: parameter array with shape (n_pixels, 10)
           [S_cloud1, deltatau1, vlos_cloud1, deltav_cloud1, loga_cloud1,
            S_cloud2, deltatau2, vlos_cloud2, deltav_cloud2, loga_cloud2]

    Returns:
        Synthetic spectrum
    
    '''
    ll, ll0 = x

    ll = torch.from_numpy(ll.astype(np.float32))
    ll0 = torch.from_numpy(ll0.astype(np.float32))
    #I_incoming = torch.from_numpy(I_incoming.astype(np.float32))

    # Same device:
    if ll.device != p.device:
        ll = ll.to(p.device)
    if ll0.device != p.device:
        ll0 = ll0.to(p.device)
    if I_incoming.device != p.device:
        I_incoming = I_incoming.to(p.device)

    # First cloud layer
    spectrum_cloud1 = cloud(p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], ll0, ll, I_incoming) 

    # Second cloud layer
    spectrum_final = cloud(p[:, 5], p[:, 6], p[:, 7], p[:, 8], p[:, 9], ll0, ll, spectrum_cloud1)

    return spectrum_final