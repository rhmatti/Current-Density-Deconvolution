#Deconvolution Script
#Author: Richard Mattish
#Last Updated: 01/05/23

#Function:  This script is used to deconvolve ion beam current data (that was
#           obtained by rastering the beam across a Faraday cup) from the
#           Faraday cup's detector function in order to obtain the ion beam
#           current density.


#Imports all of my functions and the deconvolution class I wrote
from decon_utils import *

#Creates an instance of the Deconvolution class
dec = Deconvolution()


# **********     ↓ This is where you need to define your parameters ↓     **********


dataFile = 'sample_data.txt'        #Name of data file to be imported for deconvolution
dec.Rsample = 9.7                   #Radius of the matrix (basically half of the width of the data matrix) in mm
dec.spacing = 0.3                   #Physical spacing between points in the matrix in mm
dec.R = 3.175                       #radius of aperture in mm


# **********     ↑ This is where you need to define your parameters ↑     **********


#Imports the data file as a 2D numpy array
data_matrix = np.array(processList(readFileToList(dataFile),1,0))

#Centering the largest value of the data matrix in a larger matrix and filling in all empty spaces with zeros
dec.Ic, dec.N = padnCenter(data_matrix)

#Generates the detector function file and saves it as fileName
fileName = 'Dk_test4.txt'
dec.calcDk(fileName)

#Reads in the detector function file as a matrix
dec.Dkc = processList(readFileToList(fileName),1,0)
dec.Dkc = np.array(dec.Dkc)
dec.Dk = fft.fftshift(dec.Dkc)


#Divides Ic by the area of the detector aperture
dec.Ic = dec.Ic/(np.pi*pow(dec.R,2))

#Shifts Ic from the center to create the matrix I
dec.I = fft.fftshift(dec.Ic)

#Creates the matrix Ik (the 2D discrete Fourier transform of the matrix I)
dec.Ik = fft.fft2(dec.I)

#Shifts Ik to the center to create the matrix Ikc
dec.Ikc = fft.ifftshift(dec.Ik)

#Graphs the azimuthally-averaged power spectrum of the current profile
#to aid in determining an appropriate value for eta
dec.noise2sig()

#Performs Dahl's method of iterated Wiener Filtering
dec.wienerFiltering()


#This recreates the 6 side-by-side plots that Dahl's Matlab script outputs
#(the 6 Jc plots for different values of eta)
fig, axis = plt.subplots(2,3)

p1 = axis[0,0].imshow(dec.Jcun.real, interpolation = 'gaussian', cmap = 'inferno')
fig.colorbar(p1, ax=axis[0,0])
axis[0,0].set_title(f'Jc, uniterated, eta={dec.eta}')

p2 = axis[0,1].imshow(dec.Jc_arr[0].real, interpolation = 'gaussian', cmap = 'inferno')
fig.colorbar(p2, ax=axis[0,1])
axis[0,1].set_title(f'Jc, iterated, eta={dec.scales[0]}')

p3 = axis[0,2].imshow(dec.Jc_arr[1].real, interpolation = 'gaussian', cmap = 'inferno')
fig.colorbar(p3, ax=axis[0,2])
axis[0,2].set_title(f'Jc, iterated, eta={dec.scales[1]}')

p4 = axis[1,0].imshow(dec.Jc_arr[2].real, interpolation = 'gaussian', cmap = 'inferno')
fig.colorbar(p4, ax=axis[1,0])
axis[1,0].set_title(f'Jc, iterated, eta={dec.scales[2]}')

p5 = axis[1,1].imshow(dec.Jc_arr[3].real, interpolation = 'gaussian', cmap = 'inferno')
fig.colorbar(p5, ax=axis[1,1])
axis[1,1].set_title(f'Jc, iterated, eta={dec.scales[3]}')

p6 = axis[1,2].imshow(dec.Jc_arr[4].real, interpolation = 'gaussian', cmap = 'inferno')
fig.colorbar(p6, ax=axis[1,2])
axis[1,2].set_title(f'Jc, iterated, eta={dec.scales[4]}')

plt.show()
