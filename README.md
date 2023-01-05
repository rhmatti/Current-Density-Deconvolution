# Current Density Deconvolution

Installation Instructions
--------------------
1. To run, users must have python version 3.9 or newer installed on their machine.

2. In addition to the base install of python, the following packages are required:

	a. Numpy
	
		python -m pip install numpy

	b. Matplotlib
	
		python -m pip install matplotlib

	c. Scipy
	
		python -m pip install scipy
	
3. Download the code as a .zip file, and extract all contents to the desired directory


User Instructions
--------------------
1. Open the file `Deconvolution Script.py` in your editor of choice

2. Define your input parameters:

      a. Variable `dataFile`: 
      * This is the data file containing the measured current profile of the beam
      * This data file must be formatted as a 2D matrix of evenly spaced current samples
          * The first line of the file contains the dimensions of the matrix
          * Rows correspond to y-deflection position, and columns correspond to x-deflection position
          * Numbers correspond to the measured current at that x-y position
          * See `sample_data.txt` for an example of a properly formatted data file
      * Enter the filename (if in the same directory) or full filepath (if in a different directory)
      
      b. Variable `dec.Rsample`:
      * This is the "sample radius" (i.e. half of the width of the data matrix)
      * This is a distance and should be entered in units of mm
      * This is half of the total distance the beam was deflected along its longest axis
      
      c. Variable `dec.spacing`:
      * This is the physical spacing between the data points in `dataFile` in units of mm
      
      d. Variable `dec.R`:
      * This is the radius of the detector aperture (for circular aperture detectors)
      
3. Run the script
