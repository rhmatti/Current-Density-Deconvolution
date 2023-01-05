#decon_utils
#Author: Richard Mattish
#Last Updated: 01/05/23

#Function:  This file contains all of the relevant functions to perform a deconvolution


#Importing relevant libraries
import numpy as np
import numpy.fft as fft
import scipy.special as spec
import matplotlib.pyplot as plt


#Reads all data from an input file and stores it to a list
def readFileToList(fileName):
    try:
        inputFile = open(fileName, "r")
    except(IOError):
        print("One or more of the files you entered does not exist")
        input("Press any key to exit.  ")
        quit()
    dataList = inputFile.readlines()
    return dataList


#Deletes the first i lines and last j lines from the list and parses the
#string in each entry of the list, making a "list of lists"
def processList(dataList, i, j):
    dataList2 = [] 
    while i < len(dataList)-j:
        dataList2.append(dataList[i].split())
        i = i + 1
    for row in range(0, len(dataList2)):
        dataList2[row] = [float(x) for x in dataList2[row]]
    return dataList2


#Writes a matrix to a text file
#(I use this to create the detector function file for testing with the matlab scripts)
def matrixToFile(matrix, fileName):
    xdim = matrix.shape[0]
    ydim = matrix.shape[1]
    outputFile = open(fileName, "w")
    outputFile.write(f'{xdim} {ydim}\n')
    for row in matrix:
        for entry in row:
            outputFile.write(f'{entry} ')
        outputFile.write('\n')
    outputFile.close()


#Takes a current density matrix and outputs it to a text file as a list of values
#along with the position in space of each value for plotting in other programs
def saveData(matrix, spacing, fileName):
    xdim = matrix.shape[0]
    ydim = matrix.shape[1]
    print(f'xdim={xdim}, ydim={ydim}')

    max_tuple = np.where(matrix == np.amax(matrix))
    xc = max_tuple[0][0]
    yc = max_tuple[1][0]
    print(f'xc={xc}, yc={yc}')

    outputFile = open(fileName, "w")
    outputFile.write('X-Pos\tY-Pos\tJ\n')
    yi = 0
    for row in matrix:
        ypos = (yc - yi)*spacing
        xi = 0
        for entry in row:
            xpos = (xc- xi)*spacing
            outputFile.write(f'{xpos}\t{ypos}\t{entry}\n')
            xi = xi + 1
        yi = yi + 1
    outputFile.close()


#Displays an N x M matrix as an image with N x M pixels
def showMatrix(matrix, title=None):
    plt.imshow(matrix, interpolation = 'gaussian', cmap = 'inferno', origin = 'lower')
    plt.colorbar()
    if title != None:
        plt.title(title)
    plt.show()

#Returns the number of rows and columns of a matrix
def getDim(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    return rows, columns

#Centers matrix's maximum value in the center of a larger matrix of zeros of size "size"
def padnCenter(matrix):
    ydim = matrix.shape[0]
    xdim = matrix.shape[1]
    size = 2*max([xdim, ydim])+1
    max_tuple = np.where(matrix == np.amax(matrix))
    ymax, xmax = list(zip(max_tuple[0], max_tuple[1]))[0]
    print(f'x={xmax}, y={ymax}')
    xshift = size/2-xmax
    yshift = size/2-ymax
    padded = np.zeros((size, size))
    i = 0
    while i < xdim:
        j = 0
        while j < ydim:
            padded[int(j+yshift), int(i+xshift)] = matrix[int(j), int(i)]
            j = j + 1
        i = i + 1
    return padded, size


#This class contains everything pertinent to the deconvolution calculations
class Deconvolution:
    def __init__(self):
        #Defines global variables
        self.I = None
        self.J = None

        self.Ic = None
        self.Jc = None

        self.Ik = None
        self.Dk = None
        self.Jk = None

        self.Ikc = None
        self.Dkc = None
        self.Jkc = None
        
        self.eta = None
        self.wien = None
        self.scale = None
        self.sigk = None
        self.N = None
        self.spacing = None
        self.Rsample = None
        self.R = None

    def calcDk(self, filename):
        #Creates a file for the detector function in Fourier space
        file = open(filename, "w")
        file.write(f'{self.N} {self.N}\n')

        kk = -(self.N/2)
        #Calculates Dk = 2 * J1(k*R)/(k*R) and writes to file
        while kk < (self.N/2):
            kx = 2*np.pi/(self.N*self.spacing)*kk

            jj = -(self.N/2)
            while jj < (self.N/2):
                ky = 2*np.pi/(self.N*self.spacing)*jj
                kmag = self.R*np.sqrt(kx*kx + ky*ky)
                self.Dk = 2*spec.jv(1,kmag)/kmag
                file.write(f'{round(self.Dk,7)} ')
                jj = jj + 1
            file.write('\n')
            kk = kk + 1

        file.close()

    #Computes the azimuthally-averaged power spectrum (<|I(k)|^2>) and the detector power spectrum (|D(k)|^2)
    def noise2sig(self):
        Ikc2 = pow(np.absolute(self.Ikc),2)
        Dkc2 = pow(np.absolute(self.Dkc),2)

        a = int(Ikc2.shape[0])
        b = int(Ikc2.shape[1])
        c = int((a-1)/2)

        [X, Y] = np.meshgrid(np.arange(b) - np.ceil((self.N+1)/2), np.arange(a) - np.ceil((self.N+1)/2))
        R = np.sqrt(np.square(X) + np.square(Y))
        rad = np.arange(0, np.max(R), 1)
        
        Iintensity = np.zeros(len(rad))
        Dintensity = np.zeros(len(rad))
        index = 0
        bin_size = 1

        for i in rad:
            mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
            Ivalues = Ikc2[mask]
            Iintensity[index] = np.mean(Ivalues)
            index += 1

        
        index2 = 0
        Dintensity = np.zeros(c)
        for i in range(c, a-2):
            Dintensity[index2] = Dkc2[c,i+1]
            index2 += 1
        
        Drad = np.arange(0,index2+1)
        Drad = np.arange(0,np.max(R)+1, np.max(R)/index2)

        Ioutput = open('power spectrum', 'w')
        Ioutput.write('Radius (mm^-1)\tI^2 (pA^2)\n')
        for i in range(0, len(Iintensity)):
            Ioutput.write(f'{rad[i]}\t{Iintensity[i]/np.amax(Iintensity)}\n')
        Ioutput.close

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(rad, Iintensity/np.max(Iintensity), label = '<|I$_n$(k)|$^2$>')
        #ax.plot(rad, Iintensity, label = '<|I$_n$(k)|$^2$>')
        ax.plot(Drad, Dintensity, label = '|D$_n$(k)|$^2$')

        ax.set_xlabel('k (mm$^{-1}$)')
        ax.set_ylabel('Power (A.U.)')
        ax.set_yscale('log') 
        ax.set_xscale('log')
        plt.legend()
        plt.show()
        


    #This calculates the current density by deconvolving J(k) from D(k)
    def decon(self):
        self.a = np.absolute(self.Dk)
        if self.wien == 0:
            self.den = self.a*self.a + self.eta
        elif self.wien == 2:
            self.Jka = pow(abs(self.Jkun),2)
            self.Jka = self.Jka/np.amax(self.Jka)
            self.etak = 1/self.Jka
            self.etakc = fft.ifftshift(self.etak)
            self.den = self.a*self.a + self.scale*self.etak
        self.Jk = self.Ik*self.Dk
        self.Jk = self.Jk/self.den
        self.Jkc = fft.ifftshift(self.Jk)
        self.J = fft.ifft2(self.Jk).real
        self.Jc = fft.ifftshift(self.J)

        if self.wien == 0 or self.wien == -1:
            self.Jkun = self.Jk
            self.Jkcun = fft.ifftshift(self.Jkun)
    

    #This function performs iterated Wiener Filtering
    def wienerFiltering(self):
        self.wien = 0
        self.eta = 0.01
        self.scale = 0
        self.decon()
        self.Jkcun = self.Jkc
        self.Jcun = self.Jc


        self.wien = 2
        self.mu = .6305*.758     #What is this?
        self.scales = [0.01, 0.001, 0.0001, 0.00001, 0.0004]

        self.Jkc_arr = []
        self.Jc_arr = []
        for i in range(0,len(self.scales)):
            self.scale = self.scales[i]
            self.decon()
            self.Jkc_arr.append(self.Jkc)
            self.Jc_arr.append(self.Jc)