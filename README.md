# DDM_Analysis
A repository containing the code used to analyse data from a DDM experiment. This code was used as part of a Summer Scholarship project at the University of Edinburgh.

# To Run Code:
- Code is run via calling `Main.py` in the Terminal.
- You will then be asked to input the path of the directory you wish to analyse. If you wish to analyse multiple sub-directories, input the path of a single directory containing all sub-directories to be analysied. The code assumes that the diretories to be analysed are in the same directory as the python scripts.
- You will then be prompted as to whether you wish to analyse a single directory, or compare multiple.
  
- Following the terminal prompts will allow the user to do the following operations to single directories:
  1. Display the raw data with error bars. 
  2. Display the fitted curve 
  3. Find and display the average velocity 
  4. Find and display the average Diffusion Coefficient. 
  5. Find and display the average value of A(q)
  6. Find and display the average value of the stretching exponent 
  7. Find and display the average value of tau0 in the stretched exponential ISF 
  8. Plot the fit of every q value for a given file

- If the multiple directory option is selected, the user can implement the following operations:
  1. Show the fit of each file in the directory. 
  2. Plot the average Velocity for a range of files 
  3. Plot the average Diffusion Coefficient for a range of files 
  4. Plot the average value of A(q) for a range of files 
  5. Plot the average tau0 for a range of files 
  6. Plot the average Stretching Exponent for a range of files
