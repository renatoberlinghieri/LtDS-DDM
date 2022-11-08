# LtDS-DDM
This repository contains the code used in “Measuring utility with diffusion models.” by Berlinghieri, R., Krajbich, I., Maccheroni, F., Marinacci, M., and Pirazzini, M., currently under review. 

The structure of the repository is as follows:
- The folder "Data" contains the data used in our experiments, in the desired format (.npz). These datasets represent the observed choice probabilities and decision times for the two datasets of interest, i.e. the ones used in Cavanagh et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4114997/) and Shevlin (https://www.pnas.org/content/119/6/e2101508119).
- In the folder "src" you can find the file "utils.py", that contains all the functions used in our analysis, written in Python 3. 
- The folder "Notebooks" contains two demo jupyter notebooks that show how to use the functions from utils.py to produce wavelet estimates of the DDM and the plots in our paper. One notebook covers the Cavanagh et al.'s dataset, the other the Shevlin et al.'s one. The main difference is that in the former case, the dataset is complete (i.e. every possible comparison is observed), whereas in the latter, the dataset is incomplete. We refer the interested reader to the appendix of the main paper for a thorough analysis on the topic. 
 
