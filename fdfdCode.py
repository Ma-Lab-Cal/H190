import numpy as np
import scipy as sp
from copy import deepcopy
import os
import datetime
import pickle
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sps
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, spmatrix
from scipy.sparse import dia_array, csr_array, eye, diags as diags_array
from scipy.sparse.linalg import eigs, spsolve
from scipy.sparse import identity, csr_matrix
import pickle
from fractions import Fraction
from math import gcd,ceil
from functools import reduce

def yeeder2d(ns, res, bc, kinc=None):
    """
    Derivative Matrices on a 2D Yee Grid.

    Parameters
    ----------
    ns : list of int
        Grid size [Nx, Ny].
    res : list of float
        Grid resolution [dx, dy].
    bc : list of int
        Boundary conditions [xbc, ybc].
        0: Dirichlet boundary conditions.
        1: Periodic boundary conditions.
    kinc : list of float, optional
        Incident wave vector [kx, ky].
        Required only for periodic boundary conditions.

    Returns
    -------
    dex, dey, dhx, dhy : dia_array
        Derivative matrices for electric and magnetic fields.
    """
    # Extract grid parameters
    Nx, Ny = ns
    dx, dy = res

    # Default kinc if not provided
    if kinc is None:
        kinc = [0, 0]

    # Determine matrix size
    M = Nx * Ny

    # Build DEX
    if Nx == 1:
        dex = -1j * kinc[0] * eye(M, M, format="dia")
    else:
        d0 = -np.ones(M)
        d1 = np.ones(M)
        d1[Nx::Nx] = 0
        offsets = [0, 1]
        data = np.vstack([d0, d1]) / dx
        dex = dia_array((data, offsets), shape=(M, M))

        if bc[0] == 1:
            periodic_d1 = np.zeros(M)
            periodic_d1[Nx-1::Nx] = np.exp(-1j * kinc[0] * Nx * dx) / dx
            periodic_dex = dia_array((periodic_d1[np.newaxis, :], [1 - Nx]), shape=(M, M))
            dex += periodic_dex

    # Build DEY
    if Ny == 1:
        dey = -1j * kinc[1] * eye(M, M, format="dia")
    else:
        d0 = -np.ones(M)
        d1 = np.ones(M)
        offsets = [0, Nx]
        data = np.vstack([d0, d1]) / dy
        dey = dia_array((data, offsets), shape=(M, M))

        if bc[1] == 1:
            periodic_d1 = np.exp(-1j * kinc[1] * Ny * dy) / dy * np.ones(M)
            periodic_dey = dia_array((periodic_d1[np.newaxis, :], [Nx - M]), shape=(M, M))
            dey += periodic_dey

    # Build DHX and DHY
    dhx = -dex.transpose()
    dhy = -dey.transpose()

    # Permute rows and columns to account for flattening order difference between MATLAB and numpy/scipy
    permute_indices = np.arange(M).reshape(Ny, Nx).T.flatten()
    # Convert to CSR and apply the permutation (dia_array doesn't support permutation)
    dex = csr_array(dex)[permute_indices, :][:, permute_indices]
    dey = csr_array(dey)[permute_indices, :][:, permute_indices]
    dhx = csr_array(dhx)[permute_indices, :][:, permute_indices]
    dhy = csr_array(dhy)[permute_indices, :][:, permute_indices]

    return dex, dey, dhx, dhy

def addupml2d(ER2, UR2, NPML):
    """
    Add UPML to a 2D Yee Grid.

    Parameters
    ----------
    ER2 : ndarray
        Relative permittivity on 2x grid.
    UR2 : ndarray
        Relative permeability on 2x grid.
    NPML : list or tuple of int
        [NXLO, NXHI, NYLO, NYHI] Size of UPML on 1x grid.

    Returns
    -------
    ERxx/yy/zz : ndarray
        xx/yy/zz tensor element for relative permittivity.
    URxx/yy/zz : ndarray
        xx/yy/zz tensor element for relative permeability.
    """
    # Define PML parameters
    amax = 4
    cmax = 1
    p = 3

    # Extract grid parameters
    Nx2, Ny2 = ER2.shape
    sinOrder=4
    const=6
    # Extract PML parameters
    NXLO = 2 * NPML[0]
    NXHI = 2 * NPML[1]
    NYLO = 2 * NPML[2]
    NYHI = 2 * NPML[3]

    # Initialize PML parameters to problem space
    sx = np.ones((Nx2, Ny2), dtype=complex)
    sy = np.ones((Nx2, Ny2), dtype=complex)
    # Add XLO PML
    for nx in range(1, NXLO + 1):
        ax = 1 + (amax - 1) * (nx / NXLO) ** p
        cx = cmax * np.sin(0.5 * np.pi * nx / NXLO) ** sinOrder

        # Adjust indices for Python (0-based indexing)
        sx[NXLO - nx, :] = ax * (1 - 1j * const * cx)

    # Add XHI PML
    for nx in range(1, NXHI + 1):
        ax = 1 + (amax - 1) * (nx / NXHI) ** p
        cx = cmax * np.sin(0.5 * np.pi * nx / NXHI) ** sinOrder
        sx[Nx2 - NXHI + nx - 1, :] = ax * (1 - 1j * const * cx)

    # Add YLO PML
    for ny in range(1, NYLO + 1):
        ay = 1 + (amax - 1) * (ny / NYLO) ** p
        cy = cmax * np.sin(0.5 * np.pi * ny / NYLO) ** sinOrder
        sy[:, NYLO - ny] = ay * (1 - 1j * const * cy)

    # Add YHI PML
    for ny in range(1, NYHI + 1):
        ay = 1 + (amax - 1) * (ny / NYHI) ** p
        cy = cmax * np.sin(0.5 * np.pi * ny / NYHI) ** sinOrder
        sy[:, Ny2 - NYHI + ny - 1] = ay * (1 - 1j * const * cy)

    # Incorporate PML into tensor elements
    ERxx = ER2 / sx * sy  #FLOW OF OPTPARAMS 4a
    ERyy = ER2 * sx / sy #FLOW OF OPTPARAMS 4b
    ERzz = ER2 * sx * sy #FLOW OF OPTPARAMS 4c


    URxx = UR2 / sx * sy
    URyy = UR2 * sx / sy
    URzz = UR2 * sx * sy

    # ERxx at (Ex nodes): even x (starting from index 1), odd y (starting from index 0) #FLOW OF OPTPARAMS 4d
    ERxx = ERxx[1::2, 0::2]
    # ERyy at (Ey nodes): odd x (starting from index 0), even y (starting from index 1) #FLOW OF OPTPARAMS 4e
    ERyy = ERyy[0::2, 1::2]
    # ERzz at (Ez nodes): odd x, odd y #FLOW OF OPTPARAMS 4f
    ERzz = ERzz[0::2, 0::2]

    sxsy=sx * sy
    sxosy=sx / sy
    syosx=sy / sx

    dmatx,dmaty=Nx2//2*Ny2//2,Nx2*Ny2
    dataX=syosx[1::2, 0::2].ravel()
    dataY=sxosy[0::2, 1::2].ravel()
    dataZ=sxsy[0::2, 0::2].ravel()
    rows=np.arange(0,dmatx,dtype=int)
    oneColZ=np.arange(0,Ny2,2,dtype=int)
    colsZ=[oneColZ+Ny2*i for i in range(0,Nx2,2)]
    colsZ=np.concatenate(colsZ)
    colsX=[oneColZ+Ny2*i for i in range(1,Nx2,2)]
    colsX=np.concatenate(colsX)
    oneColY=np.arange(1,Ny2,2,dtype=int)
    colsY=[oneColY+Ny2*i for i in range(0,Nx2,2)]
    colsY=np.concatenate(colsY)


    # URxx at (Hx nodes): odd x, even y
    URxx = URxx[0::2, 1::2]
    # URyy at (Hy nodes): even x, odd y
    URyy = URyy[1::2, 0::2]
    # URzz at (Hz nodes): even x, even y
    URzz = URzz[1::2, 1::2]
    return ERxx, ERyy, ERzz, URxx, URyy, URzz, [dataX,dataY,dataZ,ERxx,ERyy],rows,[colsX,colsY,colsZ]

class Structure2D:
    def __init__(self):
        self.minPerm=1.45**2
        self.maxPerm=2.05**2
        self.mode='tm'
        return

    def set_size(self,xSize,ySize,res,wl=1,nmax=3,pmlLayers=10,designRegionX=[0.2,0.8],designRegionY=[0.2,0.8],BC=[0,0],symmetrizeY=False):
        """
        Set size of the simualtion region

        Arguments:
            xSize: Scalar. x-size of simulation region in wavelengths, excluding PMLs
            ySize: Scalar. y-size of simulation region in wavelengths, excluding PMLs
            res: Int. Number of grid cells per wavelength in a region with refractive index nmax
            wl: Scalar. Wavelength in arb. units (e.g. microns)
            nmax: Scalar. Maximum refractive index in simulation region
            pmlLayers: Int. Number of PML layers
        """
        self.symY=symmetrizeY
        self.BC=BC
        self.pmlLayers=pmlLayers
        self.pmlFreq=2 * jnp.pi / wl
        if not hasattr(pmlLayers, '__len__'):
            self.NPML=[pmlLayers, pmlLayers, pmlLayers, pmlLayers]
        else:
            self.NPML=pmlLayers
        self.size=jnp.array([xSize*wl,ySize*wl])
        self.NRES=res
        self.lam0=wl
        self.k0=2*np.pi/wl
        self.nmax=nmax

        self.dx, self.dy = wl / nmax / res, wl / nmax / res
        self.Sx = xSize * self.lam0
        self.Nx = (self.NPML[0] + int(np.ceil(self.Sx / self.dx)) + self.NPML[1])//2*2
        self.Sx = self.Nx * self.dx

        self.Sy = ySize * self.lam0
        self.Ny = (self.NPML[2] + int(np.ceil(self.Sy / self.dy)) + self.NPML[3])//2*2
        self.Sy = self.Ny * self.dy
        # 2X grid
        self.Nx2 = 2 * self.Nx
        self.dx2 = self.dx / 2
        self.Ny2 = 2 * self.Ny
        self.dy2 = self.dy / 2
        self.designRegionX=[int(designRegionX[0]*self.Nx),int(designRegionX[1]*self.Nx)]
        self.designRegionY=[int(designRegionY[0]*self.Ny),int(designRegionY[1]*self.Ny)]
        self.designRegionX2=[int(designRegionX[0]*self.Nx2),int(designRegionX[1]*self.Nx2)]
        self.designRegionY2=[int(designRegionY[0]*self.Ny2),int(designRegionY[1]*self.Ny2)]
        # Calculate axis vectors
        self.xa = np.arange(1, self.Nx + 1) * self.dx
        self.ya = np.arange(1, self.Ny + 1) * self.dy
        self.xa2 = np.arange(1, self.Nx2 + 1) * self.dx2
        self.ya2 = np.arange(1, self.Ny2 + 1) * self.dy2

        # "Calculate" effective indices
        self.UR2 = np.ones((self.Nx2, self.Ny2))
        return

    def set_input(self,inXmin,inXmax,inYmin,inYmax,inModeNum,sourceRegionMin,sourceRegionMax,zeroRegion=None):
        self.inXmin=int(inXmin* self.Nx2)
        self.inXmax=int(inXmax* self.Nx2)+1
        self.inYmin=int(inYmin* self.Ny2)
        self.inYmax=int(inYmax* self.Ny2)+1
        self.inModeNum=inModeNum
        self.sourceRegionMin=int(sourceRegionMin*(self.Nx*int(inXmin==inXmax)+self.Ny*int(inYmin==inYmax)))
        self.sourceRegionMax=int(sourceRegionMax*(self.Nx*int(inXmin==inXmax)+self.Ny*int(inYmin==inYmax)))
        return

    def set_output(self,outXmins,outXmaxs,outYmins,outYmaxs,outModeNums,zeroRegion=None):
        self.outXmins=[int(outXmin* self.Nx2) for outXmin in outXmins]
        self.outXmaxs=[int(outXmax* self.Nx2)+1 for outXmax in outXmaxs]
        self.outYmins=[int(outYmin* self.Ny2) for outYmin in outYmins]
        self.outYmaxs=[int(outYmax* self.Ny2)+1 for outYmax in outYmaxs]
        self.outModeNums=outModeNums
        if zeroRegion is None:
            zeroRegion=[None,]*len(outModeNums)
        self.zeroRegion=zeroRegion
        return
    
def lcm(a, b):
    return a * b // gcd(a, b)

def gcf_of_floats(arr, max_denominator=10_000_0):
    fractions_list = [Fraction(x).limit_denominator(max_denominator) for x in arr]
    denominators = [f.denominator for f in fractions_list]
    L = reduce(lcm, denominators)
    M = [f.numerator * (L // f.denominator) for f in fractions_list]
    G = reduce(gcd, M)
    gcf_fraction = Fraction(G, L)
    return float(gcf_fraction)

def linSolver(A,b):
    return spsolve(A, b)

def inflate(perm1x):
    """
    Doubles the resolution of a given 2D matrix by repeating each element in both dimensions.

    Parameters:
        perm1x (numpy.ndarray): The input N x M permittivity matrix.

    Returns:
        numpy.ndarray: The output 2N x 2M permittivity matrix with doubled resolution.
    """
    if not isinstance(perm1x, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")

    # Repeat elements along both axes
    perm2x = np.repeat(np.repeat(perm1x, 2, axis=0), 2, axis=1)

    return perm2x

def float_gcd(a, b):
    """Compute approximate GCD of two floats by scaling to integers."""
    # Choose a scale that covers the decimal precision you need
    scale = 10**6
    A = int(round(a * scale))
    B = int(round(b * scale))
    return gcd(A, B) / scale

def gcd_of_array(arr):
    """Compute approximate GCD of a float array."""
    current = arr[0]
    for x in arr[1:]:
        current = float_gcd(current, x)
        if current == 0:
            break
    return current

def largest_dx(arr1, res, scale_power=6):
    """
    Compute the largest dx such that:
      dx <= min(arr1)/res
      and arr1[i]/dx is an integer for all i.
    """
    # 1) Approximate gcd of arr1
    arr1 = np.asarray(arr1, dtype=float)
    scale = 10**scale_power
    int_arr = [int(round(a * scale)) for a in arr1]

    # Compute integer gcd
    gcd_int = int_arr[0]
    for val in int_arr[1:]:
        gcd_int = gcd(gcd_int, val)
        if gcd_int == 0:
            break

    gcd_val = gcd_int / scale  # float gcd approximation

    # 2) Compare gcd_val to (min(arr1)/res)
    bound = np.min(arr1) / res

    if gcd_val <= bound:
        # gcd_val already fits the requirement
        return gcd_val
    else:
        # Need to scale gcd_val down by integer k >= alpha
        alpha = gcd_val / bound  # = gcd_val * res / min(arr1)
        k = ceil(alpha)
        return gcd_val / k

def resample_nearest_scipy(source, new_shape):
    """
    Resample a 2D NumPy array 'source' to 'new_shape' using
    nearest-neighbor interpolation (SciPy).
    """
    n, m = source.shape       # original shape
    N, M = new_shape          # desired new shape

    # Compute the zoom factors for each dimension
    zoom_factors = (N / n, M / m)

    # 'order=0' => nearest-neighbor
    # 'prefilter=False' avoids unnecessary filtering for nearest-neighbor
    result = scipy.ndimage.zoom(source, zoom_factors, order=0, prefilter=False)

    # zoom() can sometimes produce an array off by 1 in size due to rounding;
    # in practice it's usually correct, but if needed, we can enforce shape:
    if result.shape != (N, M):
        result = result[:N, :M]

    return result

def create_sequential_array(n, m):
    return np.arange(1, n * m + 1,dtype=int).reshape(n, m)

def modeAnalysisTM(erzz,urxx,uryy,dx,dy,k0,modeNum=0):
    erzzShape=erzz.shape
    if erzzShape[0]<erzzShape[1]:
        erzz=erzz[0,:]
    else:
        erzz=erzz[:,0]
    # Build tensor elements
    erzz = diags_array(np.squeeze(erzz), format='dia')
    urxx = diags_array(np.squeeze(urxx), format='dia')
    uryy = diags_array(np.squeeze(uryy), format='dia')
    # Build derivative matrices
    NS = erzzShape
    if erzzShape[0]<erzzShape[1]:
        RES = [1, dy]
    else:
        RES = [dx, 1]
    BC = [0, 0]
    DEX, DEY, DHX, DHY = yeeder2d(NS, k0 * np.array(RES), BC)
    # Build A and B matrices
    urxx_inv = diags_array(1 / urxx.diagonal(), format='dia')
    uryy_inv = diags_array(1 / uryy.diagonal(), format='dia')
    if erzzShape[1]<erzzShape[0]:
        A = -(DHX @ uryy_inv @ DEX + erzz)
        B = urxx_inv
    else:
        A = -(DHY @ urxx_inv @ DEY + erzz)
        B = uryy_inv
    # Solve eigenvalue problem using eigs
    num_eigenvalues = A.shape[0]-2
    ev = -3.0001**2  # Estimate of eigenvalue near which to search
    v0=np.ones(A.shape[0])
    eigenvalues, eigenvectors = eigs(A, k=num_eigenvalues, M=B, sigma=ev, which='LM',v0=v0)
    # Process eigenvalues and eigenvectors
    D = np.sqrt(eigenvalues) # D is normalized gamma; np returns principal square root, with a positive imag part
    #print(f"D: {D}")
    NEFF = -1j * D
    sort_indices = np.argsort(np.abs(NEFF.real))[::-1]  # Sort descending by real part
    NEFF = NEFF[sort_indices]
    Ez = eigenvectors[:, sort_indices]
    # Get source mode
    neff = NEFF[modeNum]
    #print(f'Effective index of input waveguide: {neff}')
    Ezsrc = Ez[:, modeNum]
    return neff, Ezsrc

def get_neffTM(stackupL,stackupN,resolution,wl,NPMLs,modeNum=0,debug=False,minSpacing='default'):
    if minSpacing=='default':
        minSpacing=0.001*wl
    maxN=np.max(stackupN)
    stackupL=np.array(stackupL)
    dxMin=np.min([wl/resolution/maxN])
    dx=np.max((gcf_of_floats(np.hstack([stackupL,dxMin])),minSpacing))
    targetNumCells=np.sum(stackupL/dx)
    cellsPerWL=wl//dx
    designRegionSize=targetNumCells/cellsPerWL
    numCellsPerLayer=stackupL/dx

    struc=Structure2D()
    struc.set_size(designRegionSize,designRegionSize/5,cellsPerWL,wl=wl,nmax=1,pmlLayers=NPMLs)
    ER2=jnp.ones((struc.Nx2,struc.Ny2),dtype=jnp.complex64)
    runningStart=struc.NPML[0]
    for layerCells,n in zip(numCellsPerLayer,stackupN):
        ER2=ER2.at[int(runningStart):int(runningStart)+int(layerCells*2),:].set(n**2)
        runningStart+=layerCells*2
    ER2=np.array(ER2)
    ERxx, ERyy, ERzz, URxx, URyy, URzz, ddata,drows,dcols = addupml2d(ER2, struc.UR2, struc.NPML)  #FLOW OF OPTPARAMS 4
    neffIn,ev=modeAnalysisTM(ER2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1],struc.UR2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1],struc.UR2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1],struc.dx,struc.dy,struc.k0,modeNum=modeNum)
    if debug:
        return neffIn, ev, ER2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1]
    else:
        return neffIn
    
def modeAnalysisTE(urzz,erxx,eryy,dx,dy,k0,modeNum=0):
    urzzShape=urzz.shape
    if urzzShape[0]<urzzShape[1]:
        urzz=urzz[0,:]
    else:
        urzz=urzz[:,0]
    # Build tensor elements
    urzz = diags_array(np.squeeze(urzz), format='dia')
    erxx = diags_array(np.squeeze(erxx), format='dia')
    eryy = diags_array(np.squeeze(eryy), format='dia')
    # Build derivative matrices
    NS = urzzShape
    if urzzShape[0]<urzzShape[1]:
        RES = [1, dy]
    else:
        RES = [dx, 1]
    BC = [0, 0]
    DEX, DEY, DHX, DHY = yeeder2d(NS, k0 * np.array(RES), BC)
    # Build A and B matrices
    erxx_inv = diags_array(1 / erxx.diagonal(), format='dia')
    eryy_inv = diags_array(1 / eryy.diagonal(), format='dia')
    if urzzShape[1]<urzzShape[0]:
        A = -(DEX @ eryy_inv @ DHX + urzz)
        B = erxx_inv
    else:
        A = -(DEY @ erxx_inv @ DHY + urzz)
        B = eryy_inv
    # Solve eigenvalue problem using eigs
    num_eigenvalues = A.shape[0]-2
    ev = -3.0001**2  # Estimate of eigenvalue near which to search
    v0=np.ones(A.shape[0])
    eigenvalues, eigenvectors = eigs(A, k=num_eigenvalues, M=B, sigma=ev, which='LM',v0=v0)
    # Process eigenvalues and eigenvectors
    D = np.sqrt(eigenvalues) # D is normalized gamma; np returns principal square root, with a positive imag part
    #print(f"D: {D}")
    NEFF = -1j * D
    sort_indices = np.argsort(np.abs(NEFF.real))[::-1]  # Sort descending by real part
    NEFF = NEFF[sort_indices]
    Hz = eigenvectors[:, sort_indices]
    # Get source mode
    neff = NEFF[modeNum]
    #print(f'Effective index of input waveguide: {neff}')
    Hzsrc = Hz[:, modeNum]
    return neff, Hzsrc

def get_neffTE(stackupL,stackupN,resolution,wl,NPMLs,modeNum=0,debug=False,minSpacing='default'):
    if minSpacing=='default':
        minSpacing=0.001*wl
    maxN=np.max(stackupN)
    stackupL=np.array(stackupL)
    dxMin=np.min([wl/resolution/maxN])
    dx=np.max((gcf_of_floats(np.hstack([stackupL,dxMin])),minSpacing))
    targetNumCells=np.sum(stackupL/dx)
    cellsPerWL=wl//dx
    designRegionSize=targetNumCells/cellsPerWL
    numCellsPerLayer=stackupL/dx

    struc=Structure2D()
    struc.set_size(designRegionSize,designRegionSize/5,cellsPerWL,wl=wl,nmax=1,pmlLayers=NPMLs)
    ER2=jnp.ones((struc.Nx2,struc.Ny2),dtype=jnp.complex64)
    runningStart=struc.NPML[0]
    for layerCells,n in zip(numCellsPerLayer,stackupN):
        ER2=ER2.at[int(runningStart):int(runningStart)+int(layerCells*2),:].set(n**2)
        runningStart+=layerCells*2
    ER2=np.array(ER2)
    ERxx, ERyy, ERzz, URxx, URyy, URzz, ddata,drows,dcols = addupml2d(ER2, struc.UR2, struc.NPML)  #FLOW OF OPTPARAMS 4
    neffIn,ev=modeAnalysisTE(struc.UR2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1],ER2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1],ER2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1],struc.dx,struc.dy,struc.k0,modeNum=modeNum)
    if debug:
        return neffIn, ev, ER2[::2,ER2.shape[1]//2:ER2.shape[1]//2+1]
    else:
        return neffIn
    
class fdfdSimObject(Structure2D):
    def __init__(self):
        """
        FDFD simulation, for interaction with SDP.

        It was supposed to be simpler than the old way...
        """
        self.pointSources={'locations':[],'radii':[],'magnitudes':[]}
        self.locations=[]
        self.faces=[]
        self.widths=[]
        self.types=[]
        self.modeNums=[]
        self.mode='tm'
        return

    def add_waveguides(self,locations,faces,widths,types,modeNums,mode='tm'):
        '''
        Adds waveguides

        Arguments:
        locations: List of floats. Centerlines of waveguides, in arb. units
        faces: List of strings. Directions of waveguides, either 'n' or 's' or 'e' or 'w'. This is the face the waveguide comes out of
        widths: List of floats. Widths of waveguides, in arb. units
        types: List of strings. Whether waveguides are 'input' or 'output'
        modeNums: List of lists of ints. Mode index to use in each waveguide, at each frequency. modeNums[i][f] is the mode number at frequency f in waveguide i. If modeNums[i] is an int, same mode will be used for all freqs
        mode: String. Either 'tm' or 'te'
        '''
        self.locations+=locations
        self.faces+=faces
        self.widths+=widths
        self.types+=types
        self.modeNums+=modeNums
        self.mode=mode.lower()
        if self.mode=='tm':
            self.modeAnalysis=modeAnalysisTM
            self.get_neff=get_neffTM
        else:
            self.modeAnalysis=modeAnalysisTE
            self.get_neff=get_neffTE
        return

    def add_pointSource(self,location,radius,magnitude=1):
        """
        Adds a point current source at the specified location. This is equivalent to an infinitely long current centered around the given location
        """
        self.pointSources['locations'].append(np.array(location))
        self.pointSources['radii'].append(radius)
        self.pointSources['magnitudes'].append(magnitude)
        return

    def add_structure(self,size,wavelengths,perms,NPMLs,padding,res,BC=[0,0],symmetry=False,snapto='wavelength-min',minLengthScale=None):
        '''
        Defines structure of the FDFD design space

        Arguments:
        size: 2-tuple of floats. The size of the simulation box, excluding PMLs, in arb. units
        wavelengths: Array-like of floats. The wavelengths, in arb. units
        perms: 2-tuple of floats. The permitivities, min and max.
        NPMLs: Float or 4-tuple of floats. If float, the number of PML layers to add on each side. If tuple, NMPLs[0]=top (min x) PMLs, NMPLs[1]=bottom (max x), NPMLs[2]=left (min y), NPMLs[3]=right (max y)
        padding: Float or 4-tuple of floats. space between PML and design region, in arb. units. If tuple, padding[0]=top (min x) padding, padding[1]=bottom (max x), padding[2]=left (min y), padding[3]=right (max y)
        res: Int. Number of cells per wavelength or structure, depending upon value of snapto.
        BC: Int or two-tuple of ints. Type of BC. 0=Dirichlet, 1=periodic. first is x-direction, second is y-direction.
        symmetry: Bool. If true, a symmetry axis will be placed in the y-direction along the bottom (max x). If you want symmetry in a different direction, tough. Rotate your coordinates.
        snapto: String. How to determine number of simulation boxes. Options are:
            'wavelength-min': The smallest wavelength in the highest permitivity region will have res cells per wavelength
            'wavelength-max': The largest wavelength in the highest permitivity region will have res cells per wavelength
            'wavelength-all': All wavelengths in the highest permitivity region will have at minimum res cells per wavelength, and all will have an integer number of cells per wavelength
            'length': The resolution will be res cells per minLengthScale
            'override-x': There will be x cells in the x-direction, INCLUDING PMLS. x must be a string integer.
        minLengthScale: Float or None. Used only if snapto=='length'
        '''
        self.size,self.wavelengths,self.perms,self.NPMLs,self.padding,self.NRES,self.BC,self.symmetry,self.snapto,self.minLengthScale=size,np.array(wavelengths),perms,NPMLs,padding,res,BC,symmetry,snapto,minLengthScale
        
        return
    
    def _add_structureShunt(self):
        if not hasattr(self.NPMLs, '__len__'):
            self.NPML=[self.NPMLs,self.NPMLs,self.NPMLs,self.NPMLs]
        if not hasattr(self.padding, '__len__'):
            self.padding=[self.padding,self.padding,self.padding,self.padding]
        self.k0s=2*np.pi/self.wavelengths
        self.nmax=np.sqrt(self.perms[1].real)
        self.nmin=np.sqrt(self.perms[0].real)
        self.minPerm,self.maxPerm=self.perms
        if self.snapto=='wavelength-min':
            wl=np.min(self.wavelengths)/self.nmax
            self.dx, self.dy = wl / self.NRES, wl / self.NRES
        elif self.snapto=='wavelength-max':
            wl=np.max(self.wavelengths)/self.nmax
            self.dx, self.dy = wl / self.NRES, wl / self.NRES
        elif self.snapto=='wavelength-all':
            wls=self.wavelengths/self.nmax
            wl=np.min(self.wavelengths)/self.nmax
            dd=largest_dx(wl,res=self.NRES)
            self.dx, self.dy = dd,dd
        elif self.snapto=='length':
            self.dx, self.dy = self.minLengthScale / self.NRES, self.minLengthScale / self.NRES
        elif self.snapto=='override-x':
            numCells=int(self.snapto.split('-')[1])
            self.dx,self.dy=self.size[0]/(self.NPMLs[0]+self.NPMLs[1]+numCells),self.size[0]/(self.NPMLs[0]+self.NPMLs[1]+numCells)

        self.Sx = self.size[0]
        self.Nx = (self.NPML[0] + int(np.ceil(self.Sx / self.dx)) + self.NPML[1])//2*2
        self.Sx = self.Nx * self.dx

        self.Sy = self.size[1]
        self.Ny = (self.NPML[2] + int(np.ceil(self.Sy / self.dy)) + self.NPML[3])//2*2
        self.Sy = self.Ny * self.dy
        self.structSize=[(self.Nx-2)//(1+int(self.symmetry)),self.Ny-2]
        # 2X grid
        self.Nx2 = 2 * self.Nx
        self.dx2 = self.dx / 2
        self.Ny2 = 2 * self.Ny
        self.dy2 = self.dy / 2
        self.designRegionX=[int(self.NPML[0]+self.padding[0]/self.dx),self.Nx-int(self.NPML[1]+self.padding[1]/self.dx)]
        self.designRegionY=[int(self.NPML[2]+self.padding[2]/self.dy),self.Ny-int(self.NPML[3]+self.padding[3]/self.dy)]
        self.designRegionX2=[int(self.NPML[0]+self.padding[0]/self.dx)*2,self.Nx2-int(self.NPML[1]+self.padding[1]/self.dx)*2]
        self.designRegionY2=[int(self.NPML[2]+self.padding[2]/self.dy)*2,self.Ny2-int(self.NPML[3]+self.padding[3]/self.dy)*2]
        self.xa = np.arange(1, self.Nx + 1) * self.dx
        self.ya = np.arange(1, self.Ny + 1) * self.dy
        self.xa2 = np.arange(1, self.Nx2 + 1) * self.dx2
        self.ya2 = np.arange(1, self.Ny2 + 1) * self.dy2
        self.UR2 = np.ones((self.Nx2, self.Ny2))
        return


    def add_objective(self,weights,targets):
        """
        Add the design objective.

        Arguments:
            weights: 2D array-like, (F, W). weights[f][i] = weight on mode overlap at frequency f in waveguide i. If i is an input, this is the target reflection
            targets: 2D array-like, (F, W). Target efficiency in waveguide i at frequency f. Currently not in use (have to either max or min overlap at a given frequency)
        """
        self.weights=weights
        self.targets=weights
        return

    def compileStructure(self):
        """
        Adds waveguides to the permitivity distribution and computes input/output modes. These are by far the most complex operations in an FDFD code.
        """
        self._add_structureShunt()
        self.fsrcs=[]
        self.ER2=np.ones((self.Nx2,self.Ny2))*self.minPerm
        self.Q = np.zeros((self.Nx, self.Ny))
        self.outputProfiles=[]

        for j,(f,mNs) in enumerate(zip(self.wavelengths,self.modeNums)):
            self.fsrcs.append(np.zeros((self.Nx, self.Ny), dtype=complex))
            self.outputProfiles.append([])
            for i,(ll,dd,ww,tt,mN) in enumerate(zip(self.locations,self.faces,self.widths,self.types,mNs)):
                if dd=='n' or dd=='s':
                    wgStartY=self.NPML[2]*2+int((ll-ww/2)/self.dy2)
                    wgEndY=self.NPML[2]*2+int((ll+ww/2)/self.dy2)+int(self.mode=='te')
                    sourceEps=np.ones((1,self.Ny2))*self.minPerm
                    sourceEps[:,wgStartY:wgEndY]=self.maxPerm
                    sourceMu=np.ones((1,self.Ny2))
                    if self.mode=='te':
                        neff,modeProfile=modeAnalysisTE(sourceMu[:,1::2],sourceEps[:,0::2],sourceEps[:,1::2],self.dx,self.dy,self.k0s[j],modeNum=mN)
                    elif self.mode=='tm':
                        neff,modeProfile=modeAnalysisTM(sourceEps[:,0::2],sourceMu[:,1::2],sourceMu[:,0::2],self.dx,self.dy,self.k0s[j],modeNum=mN)
                    if dd=='n':
                        wgStartX=None
                        wgEndX=self.NPML[0]*2+int(self.padding[0]/self.dx)*2
                        if tt=='input':
                            self.Q[self.NPML[0]+2:, :] = 1
                    elif dd=='s':
                        wgStartX=self.Nx2-self.NPML[1]*2-int(self.padding[1]/self.dx)*2
                        wgEndX=None
                        if tt=='input':
                            self.Q[:self.Nx-self.NPML[1]-2, :] = 1
                    if tt=='input':
                        for nx_i in range(self.Nx):
                            self.fsrcs[-1][nx_i, :] += modeProfile * np.exp(-1j * self.k0s[j] * neff * nx_i * self.dx)
                    self.outputProfiles[-1].append(deepcopy(modeProfile))
                elif dd=='e' or dd=='w':
                    wgStartX=self.NPML[0]*2+int((ll-ww/2)/self.dx)*2+2
                    wgEndX=self.NPML[0]*2+int((ll+ww/2)/self.dx)*2+int(self.mode=='te')
                    sourceEps=np.ones((self.Nx2,1))*self.minPerm
                    sourceEps[wgStartX:wgEndX,:]=self.maxPerm
                    sourceMu=np.ones((self.Nx2,1))
                    if self.mode=='te':
                        neff,modeProfile=modeAnalysisTE(sourceMu[1::2,:],sourceEps[1::2,:],sourceEps[0::2,:],self.dx,self.dy,self.k0s[j],modeNum=mN)
                    elif self.mode=='tm':
                        neff,modeProfile=modeAnalysisTM(sourceEps[0::2,:],sourceMu[0::2,:],sourceMu[1::2,:],self.dx,self.dy,self.k0s[j],modeNum=mN)
                    if dd=='e':
                        wgStartY=None
                        wgEndY=self.NPML[2]*2+int(self.padding[2]/self.dy)*2
                        if tt=='input':
                            self.Q[:,self.NPML[2]+2:] = 1
                    elif dd=='w':
                        wgStartY=self.Ny2-self.NPML[3]*2-int(self.padding[3]/self.dy)*2
                        wgEndY=None
                        if tt=='input':
                            self.Q[:,:self.Ny-self.NPML[3]-2] = 1
                    if tt=='input':
                        for ny_i in range(self.Ny):
                            self.fsrcs[-1][:, ny_i] = modeProfile * np.exp(-1j * self.k0s[j] * neff * ny_i * self.dy)
                    self.outputProfiles[-1].append(deepcopy(modeProfile))
                self.ER2[wgStartX:wgEndX,wgStartY:wgEndY]=self.maxPerm
        for j,(f,loc,rad,mag) in enumerate(zip(self.wavelengths,self.pointSources['locations'],self.pointSources['radii'],self.pointSources['magnitudes'])):
            self.fsrcs.append(np.zeros((self.Nx, self.Ny), dtype=complex))
            self.outputProfiles.append([])
            loc=np.asarray(loc)/np.array([self.dx,self.dy])
            loc=loc.astype(int)+np.array([self.NPML[0],self.NPML[2]])
            rad=rad//self.dx
            for nx_i in range(int(loc[0]-rad),int(loc[0]+rad)):

                for ny_i in range(int(loc[1]-rad),int(loc[1]+rad)):
                    if np.sqrt((loc[0]-nx_i)**2+(loc[1]-ny_i)**2)<rad:
                        self.fsrcs[-1][nx_i,ny_i]=1j*mag
                        self.fsrcs[-1][nx_i-1,ny_i]=1j*mag
                        self.fsrcs[-1][nx_i,ny_i-1]=1j*mag
                        self.fsrcs[-1][nx_i-1,ny_i-1]=1j*mag
        return

    def simulateStructure(self,structureMat):
        """
        Simulates a given structure, returning system matrices and paraphenalia needed for SDP and trad invDes

        Arguments:
        structureMat: Array-like. This is the permitivity distribution in the design region, defined by self.designRegionX2 and self.designRegionY2. If not the correct size, will be nearest-neighbor interpolated.

        Returns:
        returnVals: List of length F. One entry per frequency. returnVals[F] has the structure:
            [A,b,nodeMap,[dERdER2_data, dERdER2_rows, dERdER2_cols],[DEX, DEY, DHX, DHY, ERxx_inv, ERyy_inv],outputModes]
        """
        useER2=deepcopy(self.ER2)
        useER2[self.designRegionX2[0]:self.designRegionX2[1],self.designRegionY2[0]:self.designRegionY2[1]]=resample_nearest_scipy(structureMat,(self.designRegionX2[1]-self.designRegionX2[0],self.designRegionY2[1]-self.designRegionY2[0]))
        self.useER2=useER2
        ERxx, ERyy, ERzz, URxx, URyy, URzz, ddata,drows,dcols = addupml2d(useER2, self.UR2, self.NPML)  #FLOW OF OPTPARAMS 4
        ERxx = diags_array(ERxx.ravel(), format='dia')  #FLOW OF OPTPARAMS 5
        ERyy = diags_array(ERyy.ravel(), format='dia')
        ERzz = diags_array(ERzz.ravel(), format='dia')
        URxx = diags_array(URxx.ravel(), format='dia')
        URyy = diags_array(URyy.ravel(), format='dia')
        URzz = diags_array(URzz.ravel(), format='dia')
        NS = [self.Nx, self.Ny]
        RES = [self.dx, self.dy]
        BC = self.BC
        returnVs=[]
        for j,(freq,fsrc,k0,oMs) in enumerate(zip(self.wavelengths,self.fsrcs,self.k0s,self.outputProfiles)):
            DEX, DEY, DHX, DHY = yeeder2d(NS, k0 * np.array(RES), BC)
            if self.symmetry:
                quarter=DEX.shape[0]//2
            else:
                quarter=DEX.shape[0]
            ERyy_inv = diags_array(1 / ERyy.diagonal()[:quarter], format='dia')
            ERxx_inv = diags_array(1 / ERxx.diagonal()[:quarter], format='dia')
            DEX=DEX[:quarter,:quarter]
            DEY=DEY[:quarter,:quarter]
            DHY=DHY[:quarter,:quarter]
            DHX=DHX[:quarter,:quarter]
            URyy_inv = diags_array(1 / URyy.diagonal()[:quarter], format='dia')
            URxx_inv = diags_array(1 / URxx.diagonal()[:quarter], format='dia')
            ERzz=diags_array(ERzz.diagonal()[:quarter], format='dia')
            URzz=diags_array(URzz.diagonal()[:quarter], format='dia')
            if self.symmetry and self.mode=='tm':
                DEX[-self.Ny:]=0
            if self.mode=='tm':
                A = DHX @ URyy_inv @ DEX + DHY @ URxx_inv @ DEY + ERzz
            else:
                A = DEX @ ERyy_inv @ DHX + DEY @ ERxx_inv @ DHY + URzz
            self.ERzz=ERzz
            fsrc = fsrc.ravel()[:quarter]
            # Calculate scattered-field masking matrix
            #Q = diags_array(self.Q.ravel()[:quarter], format='dia')
            # Calculate source vector
            b = fsrc#(Q @ A - A @ Q) @ fsrc
            returnVs.append([A,b])
            nodeMap=np.pad(create_sequential_array(self.Nx-2,self.Ny-2),1,"constant",constant_values=0).ravel()
            nodeMap=nodeMap[:quarter]
            self.quarter=quarter
            returnVs[-1].append(nodeMap)
            returnVs[-1].append((ddata,drows,dcols))
            if self.mode=='te':
                returnVs[-1].append([DEX,DEY,DHX,DHY,ERxx_inv,ERyy_inv])
            else:
                returnVs[-1].append([DEX,DEY,DHX,DHY,URxx_inv,URyy_inv])
            returnVs[-1].append(oMs)
        return returnVs


    def compileSDP(self):
        self.sizeOneFreq=self.Nx*self.Ny*2//(1+int(self.symmetry))
        self.grandASize=[self.sizeOneFreq*len(self.wavelengths),self.sizeOneFreq*len(self.wavelengths)]

        self.grandAUp=np.zeros(self.grandASize)
        structMat=np.ones((self.designRegionX2[1]-self.designRegionX2[0],self.designRegionY2[1]-self.designRegionY2[0]))*self.maxPerm
        rvs=self.simulateStructure(structMat)
        for i,rvSet in enumerate(rvs):
            A,b=rvSet[0].todense(),rvSet[1]
            saR=np.block([[np.real(A),-np.imag(A)],[np.imag(A),np.real(A)]])
            self.grandAUp[self.sizeOneFreq*i:self.sizeOneFreq*(i+1),self.sizeOneFreq*i:self.sizeOneFreq*(i+1)]=saR

        self.grandNodeMap=[]
        self.grandb=np.zeros(self.sizeOneFreq*len(self.wavelengths))
        self.grandADown=np.zeros(self.grandASize)
        structMat=np.ones((self.designRegionX2[1]-self.designRegionX2[0],self.designRegionY2[1]-self.designRegionY2[0]))*self.minPerm
        rvs=self.simulateStructure(structMat)
        for i,rvSet in enumerate(rvs):
            A,b=rvSet[0].todense(),rvSet[1]
            sbR=np.block([[np.real(A),-np.imag(A)],[np.imag(A),np.real(A)]])
            self.grandADown[self.sizeOneFreq*i:self.sizeOneFreq*(i+1),self.sizeOneFreq*i:self.sizeOneFreq*(i+1)]=sbR
            self.grandNodeMap+=list(rvSet[2])+list(rvSet[2])
            self.grandb[self.sizeOneFreq*i:self.sizeOneFreq*(i+1)]=np.hstack([np.real(b),np.imag(b)])

        self.sdp=SDPCore()
        self.sdp.system2SDP(self.grandADown,self.grandAUp,self.grandb,nodeMap=np.array(self.grandNodeMap),zeroFloor=1E-12)
        return self.sdp

    def a2v(self,coords):
        return coords[0]*self.Ny+coords[1]

    def compileA0(self):
        self.A0=np.zeros(self.sdp.constraints[0].shape)
        complexShift=self.sizeOneFreq//2
        for j,(weights,targets,OMPs) in enumerate(zip(self.weights,self.targets,self.outputProfiles)):
            runningShift=self.sizeOneFreq*j
            for i, (weight,target,OMP,ff,ll,ww) in enumerate(zip(weights,targets,OMPs,self.faces,self.locations,self.widths)):
                if ff=='n':
                    outlinIndices=[self.a2v((self.NPML[0]+1,yi)) for yi in range(self.Ny)]
                elif ff=='s':
                    outlinIndices=[self.a2v((self.Nx-self.NPML[1]-1,yi)) for yi in range(self.Ny)]
                elif ff=='e':
                    outlinIndices=[self.a2v((xi,self.NPML[2]+1)) for xi in range(self.Nx)]
                elif ff=='w':
                    outlinIndices=[self.a2v((xi,self.Ny-self.NPML[3]-1)) for xi in range(self.Nx)]
                outlinIndices=np.array(outlinIndices)
                if self.symmetry:
                    outlinIndices=outlinIndices[:len(outlinIndices)//2]
                for spock,(ind1,eTarget1) in enumerate(zip(outlinIndices,OMP)):
                    for ind2,eTarget2 in zip(outlinIndices[spock:],OMP[spock:]):
                        mij=np.conjugate(eTarget1)*eTarget2

                        self.A0[ind1+runningShift,ind2+runningShift]=weight*mij.real
                        self.A0[ind1+runningShift+complexShift,ind2+runningShift+complexShift]=weight*mij.real
                        self.A0[ind1+runningShift,ind2+runningShift+complexShift]=weight*mij.imag
                        self.A0[ind1+runningShift+complexShift,ind2+runningShift]=weight*mij.imag
        return self.A0

    def getSO(self,structureMat):
        if self.symmetry:
            structureMat=np.pad(np.vstack((structureMat,np.flip(structureMat,axis=0))),(1,1),mode='constant')
        structureMat=structureMat*(self.maxPerm-self.minPerm)+self.minPerm
        structureMat=structureMat[self.designRegionX[0]-int(self.symmetry):self.designRegionX[1]+int(self.symmetry),self.designRegionY[0]:self.designRegionY[1]]
        rVs=self.simulateStructure(structureMat)
        '''
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(4,4))
        ax.imshow(np.where(self.usetMap<0,structureMat,self.usetMap),cmap='Greys')
        plt.show()
        '''
        xvec=[]
        sparams=[]
        for j,(rvSet,OMPs) in enumerate(zip(rVs,self.outputProfiles)):
            sparams.append([])
            A,b=rvSet[0],rvSet[1]
            evec=linSolver(A,b)
            xvec+=list(evec.real)+list(evec.imag)
            x=np.reshape(evec,(self.Nx//(1+int(self.symmetry)),self.Ny))
            if self.symmetry:
                x=np.vstack([x,np.flip(x,axis=0)])
            for i, (OMP,ff,ll,ww) in enumerate(zip(OMPs,self.faces,self.locations,self.widths)):
                if ff=='n':
                    outSlice=x[self.NPML[0]+1,:]
                elif ff=='s':
                    outSlice=x[self.Nx-self.NPML[1]-1,:]
                elif ff=='e':
                    outSlice=x[:,self.NPML[2]+1]
                elif ff=='w':
                    outSlice=x[:,self.Ny-self.NPML[3]-1]
                sparams[-1].append(np.abs(np.sum(np.conjugate(outSlice)*OMP))**2)
        xvec=np.hstack((np.array(xvec),np.array((1))))
        return np.trace(self.A0 @ np.outer(xvec,xvec)),sparams