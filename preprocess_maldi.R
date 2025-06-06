###############################################################################
##
## MALDI-TOF Translator from Bruker to mzML
##
## Alejandro Guerrero-López
##
###############################################################################
args <- commandArgs(trailingOnly = TRUE)
# First argument: path to the folder containing the raw data
# Second argument: path to the folder where the processed data will be stored
# Third argument: 1 if the data contains replicates, 0 otherwise


###############################################################################
## Load libraries
## MALDIquant()
## MALDIquantForeign()
##
dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)  # create personal library
.libPaths(Sys.getenv("R_LIBS_USER"))  # add to the path
# install.packages(c("MALDIquant","MALDIquantForeign"), repos = "http://cran.us.r-project.org")
# install.packages("stringi", dependencies=TRUE, INSTALL_opts = c('--no-lock'), repos = "http://cran.us.r-project.org")
# install.packages("stringr", dependencies=TRUE, INSTALL_opts = c('--no-lock'), repos = "http://cran.us.r-project.org")
##############################################################################

library("MALDIquant")
library("MALDIquantForeign")
library("stringr")

###############################################################################
## Load data
###############################################################################
# replicates <- args[3]
# if (replicates>0) {id_pos<-5} else {id_pos<-4}

path_train <- args[1]
path_export <- args[2]

print(path_train)
print(path_export)

sprintf("Loading MALDI raw data...")

spectra1 <- importBrukerFlex(path_train)

##### PREPROCESS

sprintf("Preprocessing MALDI raw data...")
#Step 1: the measured intensity is transformed with a square-root method to stabilize the variance
spectraint <- transformIntensity(spectra1, method="sqrt")
# Step 2: smoothing using the Savitzky–Golay algorithm with half-window-size 5 is applied
spectrasmooth <- smoothIntensity(spectraint, method="SavitzkyGolay", halfWindowSize=5, polynomialOrder=3)
# Step 3: an estimate of the baseline
spectrabase <- removeBaseline(spectrasmooth, method="TopHat")
# Step 4: replicates handling
# if (replicates>0){
# id_idx <- length(str_split(metaData(spectrabase[[1]])$file, "/", simplify=TRUE))-id_pos
# samples <- factor(sapply(sapply(spectrabase, function(x)metaData(x)$file), function(x)str_split(x, "/", simplify=TRUE))[,id_idx])
# avgSpectra <- averageMassSpectra(spectrabase, labels=samples, method="mean")
# } else { avgSpectra <- spectrabase
# }
# Step 5: alignment
#spectra_al <- alignSpectra(avgSpectra, halfWindowSize=20, SNR=2, tolerance=600e-6, warpingMethod="lowess")
# Step 6: the intensity is calibrated using the total ion current (TIC)
spectra_tic <- calibrateIntensity(spectrabase, method="TIC")

sprintf("MALDI RAW data preprocessed.")
sprintf("Storing MALDI processed data...")
###############################################################################
## Save data
###############################################################################
# path_save <- paste0(path, args[2])
# save(spectra1, file=args[2])
## Export
# exportMzMl(spectra_tic, path=path_save)
for (i in c(1:length(spectra_tic))) {
    export(spectra_tic[i], file=paste(path_export, str_split(metaData(spectra_tic[[i]])$file, "/", simplify=TRUE)[,10], sep=""), type="csv", force=TRUE)
}
sprintf("MALDI processed data stored at %s", path_export)
