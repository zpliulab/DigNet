
library(SAVER, lib.loc = '/home/wcy/miniconda3/envs/discrete-diffusion/lib/R/library')

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  filepath <- '/home/wcy/Diffusion/CancerDatasets/DCA/BRCA_input.csv'
}else{
  filepath <- args[1]
}

exp <- read.csv(filepath)
rownames(exp) <- exp[,1]
exp <- exp[,-1]
exp_saver <- saver(exp, ncores = 64, estimates.only = TRUE)


out_filepath <- gsub("input", "output", filepath)
write.csv(exp_saver,file = out_filepath)
