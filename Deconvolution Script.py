#Deconvolution Script

#Author: Richard Mattish
#Adapted from Eric Dahl's matlab code (DOI: https://doi.org/10.1063/1.1287635)

#Last Updated: 06/18/22

#Function:  This script is used to deconvolve ion beam current data (that was
#           obtained by rastering the beam across a Faraday cup) from the
#           Faraday cup's detector function in order to obtain the ion beam
#           current density.



#Importing relevant libraries
from cProfile import label
import numpy as np
import numpy.fft as fft
import scipy.signal as sig
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

#Takes a matrix and separates columns as distinct numpy arrays
def processData(dataList):
    I = []
    XV = []
    YV = []
    for row in dataList:
        I.append(row[1])
        XV.append(row[2])
        YV.append(row[3])
    I = np.array(I)
    XV = np.array(XV)
    YV = np.array(YV)
    return XV, YV, I

#Returns the location of all peaks (since my data was collected in pulsed mode)
def findPeaks(x, y, z, Vmax, Vstep):
    peaks = np.zeros((int(2*Vmax/Vstep+1), int(2*Vmax/Vstep+1)))
    for i in range(-Vmax,Vmax+Vstep, Vstep):
        x_matches = np.where(x == i)[0]
        y1 = y[x_matches]
        z1 = z[x_matches]
        for j in range(-Vmax,Vmax+Vstep, Vstep):
            y_matches = np.where(y1 == j)[0]
            z2 = z1[y_matches]
            peaks[int((i+Vmax-2)/Vstep),int((j+Vmax-2)/Vstep)] = np.amax(z2)
    return peaks

#Returns two arrays representing x and y slices at the coordinates of the maximum value of z
def findCenterSlices(x, y, z):
    imax = np.argmax(z)
    xc = x[imax]
    yc = y[imax]
    print(f'Maximum located at: ({xc},{yc})')
    xslice = []
    yslice = []
    for i in range(0, len(x)):
        if x[i] == xc:
            yslice.append([y[i], z[i]])
        if y[i] == yc:
            xslice.append([x[i], z[i]])
    xslice = np.array(xslice)
    yslice = np.array(yslice)
    return xslice, yslice

#Converts deflector voltage difference to deflection in mm
def convertVtoD(Voltages):
    q = 6*1.6*pow(10,-19)
    m = 16*1.66*pow(10,-27)
    d = 0.0254
    dy1 = 0.0254
    dy2 = 0.4826
    r = 0.35
    B = 35.7*pow(10,-3)
    viy = q*B*r/m
    a = Voltages*q/(m*d)
    dx1 = 0.5*a*pow(dy1/viy,2)
    vfx = a*dy1/viy
    dx2 = dx1 + vfx*dy2/viy
    dx2 = dx2*1000
    
    return dx2


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
    plt.imshow(matrix, interpolation = 'gaussian', cmap = 'inferno')
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
def padnCenter(matrix, size):
    xdim = matrix.shape[0]
    ydim = matrix.shape[1]
    max_tuple = np.where(matrix == np.amax(matrix))
    xmax, ymax = list(zip(max_tuple[0], max_tuple[1]))[0]
    xshift = size/2-xmax
    yshift = size/2-ymax
    padded = np.zeros((size, size))
    i = xshift
    while i < (size-(xdim-xshift)):
        j = yshift
        while j < (size-(ydim-yshift)):
            padded[int(i),int(j)] = matrix[int(i-xshift), int(j-yshift)]
            j = j + 1
        i = i + 1
    return padded

#Shifts a matrix to the center from the origin
def shifttoc(matrix):
    xdim = matrix.shape[0]
    ydim = matrix.shape[1]
    if xdim == 0 or ydim == 0 or xdim != ydim:
        print('Error')
        return
    else:
        n = int(xdim/2)
        s = np.concatenate((np.arange(n+1,xdim),np.arange(0,n+1)))
        m = matrix[np.ix_(s,s)]
    return m

#Shifts a matrix from the center to the origin
def shiftfrc(matrix):
    xdim = matrix.shape[0]
    ydim = matrix.shape[1]
    if xdim == 0 or ydim == 0 or xdim != ydim:
        print('Error')
        return
    else:
        n = int(xdim/2)
        s = np.concatenate((np.arange(n,xdim),np.arange(0,n)))
        m = matrix[np.ix_(s,s)]
    return m


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
        self.theta = None
        self.N = None
        self.spacing = None
        self.Rsample = None
        self.Nint = None
        self.th = None
        self.energy = None
        self.npoints = None
        self.delta = None
        self.R = None

    def calcDk(self, filename):
        #Creates a file for the detector function in Fourier space
        file = open(filename, "w")
        file.write(f'{self.npoints} {self.npoints}\n')

        kk = -(self.npoints/2)
        #Calculates Dk = 2 * J1(k*R)/(k*R) and writes to file
        while kk < (self.npoints/2):
            kx = 2*np.pi/(self.npoints*self.delta)*kk

            jj = -(self.npoints/2)
            while jj < (self.npoints/2):
                ky = 2*np.pi/(self.npoints*self.delta)*jj
                kmag = self.R*np.sqrt(kx*kx + ky*ky)
                self.Dk = 2*spec.jv(1,kmag)/kmag
                file.write(f'{round(self.Dk,7)} ')
                jj = jj + 1
            file.write('\n')
            kk = kk + 1

        file.close()


    #Calculates a matrix representation of the cross section of the detector as seen by the ion beam
    #at a given angle of incidence, theta
    def calcth(self):
        Ncenter = np.ceil((self.Nint+1)/2)
        thetar = self.theta*np.pi/180
        fx = 1/pow(np.cos(thetar),2)
        print(f'cos({self.theta}) = {np.cos(thetar)}')
        print(f'scaling factor = {fx}')

        lim2 = pow((self.Rsample/self.spacing),2)

        self.th = np.zeros((self.Nint,self.Nint))

        for i in range(1,self.Nint):
            for j in range(1,self.Nint):
                if pow((i-Ncenter),2) + fx*pow((j-Ncenter),2) <= lim2:
                    self.th[i-1,j-1]=1

    #Computes the azimuthally-averaged power spectrum (<|I(k)|^2>) and the detector power spectrum (|D(k)|^2)
    def sig2noise(self):
        Ikc2 = pow(np.absolute(self.Ikc),2)
        Dkc2 = pow(np.absolute(self.Dkc),2)

        a = Ikc2.shape[0]
        b = Ikc2.shape[1]

        [X, Y] = np.meshgrid(np.arange(b) - np.ceil((self.Nint+1)/2), np.arange(a) - np.ceil((self.Nint+1)/2))
        R = np.sqrt(np.square(X) + np.square(Y))
        rad = np.arange(0, np.max(R), 1)
        
        Iintensity = np.zeros(len(rad))
        Dintensity = np.zeros(len(rad))
        index = 0
        bin_size = 1

        for i in rad:
            mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
            Ivalues = Ikc2[mask]
            Dvalues = Dkc2[mask]
            Iintensity[index] = np.mean(Ivalues)
            Dintensity[index] = np.mean(Dvalues)
            index += 1

        index2 = 0
        Dintensity = np.zeros(65)
        for i in range(65, 129):
            Dintensity[index2] = Dkc2[65,i]
            index2 += 1

        Drad = np.arange(0,index2+1)
        Drad = np.arange(0,np.max(R)+1, np.max(R)/index2)
        

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
        


    #This is a recreation of Dahl's matlab file "decon2.m" for wien=0 and wien=2
    def decon(self):
        self.a = np.absolute(self.Dk)
        if self.wien == 0:
            self.den = self.a*self.a + self.eta
        elif self.wien == 2:
            self.Jka = pow(abs(self.Jkun),2)
            self.Jka = self.Jka/np.amax(self.Jka)
            self.etak = 1/self.Jka
            self.etakc = shifttoc(self.etak)
            self.den = self.a*self.a + self.scale*self.etak
        self.Jk = self.Ik*self.Dk
        self.Jk = self.Jk/self.den
        self.Jkc = shifttoc(self.Jk)
        self.J = fft.ifft2(self.Jk).real
        self.Jc = shifttoc(self.J)

        if self.wien == 0 or self.wien == -1:
            self.Jkun = self.Jk
            self.Jkcun = shifttoc(self.Jkun)

    #This is an almost one-to-one recreation of Dahl's matlab file "decon2.m"
    def decon2(self):
        self.a = np.absolute(self.Dk)

        if self.wien == 1:
            Nhalf = int(self.N/2)
            [x,y] = np.meshgrid(np.arange(-Nhalf,Nhalf))
            #Other stuff goes here

        elif self.wien == 2 or self.wien == -2:
            self.Jka = pow(abs(self.Jkun),2)
            self.Jka = self.Jka/np.amax(self.Jka)
            self.etak = 1/self.Jka
            self.etakc = shifttoc(self.etak)

        if self.wien >= 0:
            if self.wien == 0:
                self.den = self.a*self.a + self.eta
            elif self.wien > 0:
                self.den = self.a*self.a + self.scale*self.etak
            self.Jk = self.Ik*self.Dk
            self.Jk = self.Jk/self.den

        elif self.wien < 0:
            if self.wien == -1:
                self.den = np.sqrt(pow(self.a,2) + self.eta)
            elif self.wien < -1:
                self.den = np.sqrt(pow(self.a,2) +self.scale*self.etak)
        
        self.Jkc =shifttoc(self.Jk)
        self.J = fft.ifft2(self.Jk).real
        self.Jc = shifttoc(self.J)

        if self.wien == 0 or self.wien == -1:
            self.Jkun = self.Jk
            self.Jkcun = shifttoc(self.Jkun)

    #Calculates integral J, integral J2, and their ratio (intJ2/intJ)
    def calcints(self):
        Jcth = self.th*self.Jc
        Jcth2 = Jcth*Jcth
        Nsum = sum(sum(self.th))
        thetar = self.theta*np.pi/180
        costh = np.cos(thetar)
        Area = np.pi*pow(self.Rsample,2)*costh
        print(f'Sum={sum(sum(Jcth))}')

        self.intJ = sum(sum(Jcth))/Nsum*Area
        self.intJ2 = sum(sum(Jcth2))/Nsum*Area*costh
        self.ratio = self.intJ2/self.intJ
        print(f'{self.intJ}\t{self.intJ2}\tratio={self.ratio}')


    def calcresi(self):
        results = open('results.txt', 'a')
        results.write(f'{self.energy}\t{self.theta}\t{self.N}\t{self.wien}\t{self.eta}\t{self.scale}\t{self.sigk}\
            \t{self.Nint}\t{self.intJ}\t{self.intJ2}\t{self.ratio}\n')
        results.close()


    #This is the order of functions that are called repeatedly by wienerFiltering for each value of eta
    def cycle(self):
        self.decon()    #You can also use self.decon2() here
        self.theta = 0
        self.calcth()
        self.calcints()
        self.calcresi()
    

    #This function performs iterated Wiener Filtering according to Dahl's method
    def wienerFiltering(self):
        results = open('results.txt', 'w')
        results.write(f'energy\ttheta\tN\twien\teta\tscale\tsigk\
            \tNint\tintJ\tintJ2\tratio\n')
        results.close()
        index = 1
        self.wien = 0
        self.eta = 0.01
        self.scale = 0
        self.cycle()
        self.Jkcun = self.Jkc
        self.Jcun = self.Jc


        self.wien = 2
        self.mu = .6305*.758     #What is this?
        self.scales = [0.01, 0.001, 0.0001, 0.00001, 0.0004]

        self.Jkc_arr = []
        self.Jc_arr = []
        for i in range(0,len(self.scales)):
            self.scale = self.scales[i]
            self.cycle()
            self.Jkc_arr.append(self.Jkc)
            self.Jc_arr.append(self.Jc)

            max_Jc = np.amax(self.Jc)*self.mu
            Nhalf = int(self.N/2)+1
            c_valueJc = self.Jc[Nhalf,Nhalf]*self.mu
            average_20mils = 0.25*(self.Jc[Nhalf-4,Nhalf]+self.Jc[Nhalf+4,Nhalf]+self.Jc[Nhalf,Nhalf-4]+self.Jc[Nhalf,Nhalf+4])


#Creates an instance of the Deconvolution class
dec = Deconvolution()

#Need to define variables for inputs to the code
dec.energy = 4500  #energy of the ion beam
dec.npoints =129   #number of points per side
dec.Rsample = convertVtoD(128)      #Radius of the matrix (basically half of the width of the data matrix) in mm
dec.spacing = convertVtoD(4)        #Physical spacing between points in the matrix in mm
print(f'Rsample={dec.Rsample}')
print(f'spacing={dec.spacing}')
dec.theta = 0
dec.N = dec.npoints
dec.Nint = dec.N
dec.th = dec.calcth()
dec.R = 3       #radius of aperture in mm
dec.delta = 0.351   #Point spacing in mm

#Generates the detector function file and saves it as fileName
fileName = 'Dk_test4.txt'
dec.calcDk(fileName)

#Reads in the detector function file as a matrix
dec.Dkc = processList(readFileToList(fileName),1,0)
dec.Dkc = np.array(dec.Dkc)
dec.Dk = shiftfrc(dec.Dkc)
print(f'Dk matrix is {dec.Dk.shape}')
max_tuple = np.where(dec.Dk == np.amax(dec.Dk))
ListofCoordinates = list(zip(max_tuple[0], max_tuple[1]))
for coord in ListofCoordinates:
    print(f'Maximum at {coord}')


#These four lines are inporting and processing the 4 different data files I took
#Each file represents 1 quadrant of space that I rastered the ion beam over the Faraday cup
XV1, YV1, I1 = processData(processList(readFileToList('deflector_quad1.log'),1,0))
XV2, YV2, I2 = processData(processList(readFileToList('deflector_quad2.log'),1,0))
XV3, YV3, I3 = processData(processList(readFileToList('deflector_quad3.log'),1,0))
XV4, YV4, I4 = processData(processList(readFileToList('deflector_quad4.log'),1,0))


#I am combining the 4 different files into 3 arrays 
#(X Deflector Voltage, Y Deflector Voltage, and Current)
XV = np.concatenate((XV1, XV2, -XV3, -XV4))
YV = np.concatenate((YV1, -YV2, YV3, -YV4))
I = np.concatenate((I1, I2, I3, I4))


#Finding just the current peaks (since the data was pulsed) and returning as a 64x64 matrix
peak_matrix = findPeaks(XV, YV, I, 64, 2)

#Centering the largest value of the 64x64 matrix in a larger 129x129 matrix and filling in all empty spaces with zeros
peak_matrix = padnCenter(peak_matrix, 129)
showMatrix(peak_matrix)

print(f'Peak matrix is {peak_matrix.shape}')
max_tuple = np.where(peak_matrix == np.amax(peak_matrix))
ListofCoordinates = list(zip(max_tuple[0], max_tuple[1]))
for coord in ListofCoordinates:
    print(f'Maximum at {coord}')

#Defines the matrix Ic (aka the peak current matrix that I took experimentally)
#Basically just renaming it to match the math equations
dec.Ic = peak_matrix

#Divides Ic by the area of the detector aperture
dec.Ic = dec.Ic/(np.pi*pow(dec.R,2))

#Shifts Ic from the center to create the matrix I
dec.I = shiftfrc(dec.Ic)

#Creates the matrix Ik (the 2D discrete Fourier transform of the matrix I)
dec.Ik = fft.fft2(dec.I)

#Shifts Ik to the center to create the matrix Ikc
dec.Ikc = shifttoc(dec.Ik)


dec.sig2noise()

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


#Saves dec.Jc_arr[0].real as a file to be used for plotting in other software (for me, Mathematica)
saveData(dec.Jc_arr[0].real, dec.spacing, 'Jc eta=0.01 iterated data')



#This creates a plot of the data I took in 3D without deconvolution
#(so you can see what I actually measured)
'''
print(len(I))
peak_indices = sig.find_peaks(I)[0]
peak_I = I[peak_indices]
peak_XV = XV[peak_indices]
peak_YV = YV[peak_indices]
print(len(peak_I))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(peak_XV, peak_YV, peak_I)
ax.set_xlabel('X Deflector (V)')
ax.set_ylabel('Y Deflector (V)')
ax.set_zlabel('Ion Beam Current (pA)')
plt.show()
'''
