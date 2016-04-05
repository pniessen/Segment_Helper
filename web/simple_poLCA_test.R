# simple poLCA tester - to be run via Python wrapper
# 1.25.2016 Peter Niessen
#
#infile <- "/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/sample_case_work/GoPro/X_rebucketed.csv"
#infile <- "/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/sample_case_work/GoPro/X_rebucketed_3.csv"
#ds = read.csv("/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/sample_case_work/GoPro/X_rebucketed_2.csv")
#'/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/sample_case_work/GoPro/X_rebucketed_2.csv'
#ds = read.csv(infile)

library(poLCA)
library(weights)

myArgs <- commandArgs(trailingOnly = TRUE)
myArgs2 <- unlist(myArgs)
print (myArgs2)
basedir <-  toString(myArgs2[3])
infile <- toString(myArgs2[4])
timestamp <-toString(myArgs2[5])
vars <- toString(myArgs2[7:length(myArgs2)-1])
weights_filename <- toString(myArgs2[length(myArgs2)])
#vars <- c("A","B","C","D")
#vars <- c('q06_1', 'q06_2', 'q06_3','q06_4','q06_5','q06_6','q06_7')
print (vars)
print (weights_filename)
#f1 <- paste0(basedir,"predicted_segment.txt")
#if (file.exists(f1)) file.remove(f1)
#f2 <- paste0(basedir,"posterior_probabilities.txt")
#if (file.exists(f2)) file.remove(f2)
#f3 <- paste0(basedir, "model_stats.txt")
#if (file.exists(f3)) file.remove(f3)

ds = read.csv(paste0(basedir,infile))

#SegFn <- as.formula(paste0("cbind(",var
#model_function <- cbind(A,B,C,D)~1
#model_function <- cbind(vars)~1
model_function <- as.formula(paste0("cbind(",toString(vars),")~1"))

print (model_function)

#model function
#cbind(q32_1, q32_2, q32_3, q35_7, q35_1, q35_8, q35_6, q48_7, 
#    q48_27, q48_13, q35_2, q35_11, q35_3, q39_4, q48_26, q48_24, 
#    q36, q34_2_Bucketed) ~ 1
num_seg <- as.numeric(myArgs2[1])
print(num_seg)
#maxiter <- c(5000)
nrep <- myArgs2[2]  

#print (vars)

res2 = poLCA(model_function, maxiter=50000, nclass=num_seg, nrep=nrep, data=ds)
#res2 = poLCA(cbind(A,B,C,D) ~ 1, maxiter=50000, nclass=3, nrep=10, data=ds)
# name dataframe

# create combined matrix to filter to calc mean posterior probabilities
posterior_probabilities <- res2$posterior
predicted_segment <- res2$predclass
model_mle <- res2$llik
model_chisq <- res2$Chisq
model_aic <- res2$aic
model_bic <- res2$bic
seg_and_proba <- cbind(posterior_probabilities, predicted_segment)
model_stats <- c(model_mle, model_chisq, model_bic)

# write to disk to handoff to python
write.table(predicted_segment, file = paste0(basedir,paste0(paste0("/static/model/predicted_segment_",timestamp),".txt")), sep=",", col.names = F,row.names = F)
write.table(posterior_probabilities, file = paste0(basedir,paste0(paste0("/static/model/posterior_probabilities_",timestamp),".txt")), sep=",", col.names = F,row.names = F)
write.table(model_stats, file = paste0(basedir,paste0(paste0("/static/model/model_stats_",timestamp),".txt")), sep=",", col.names = F,row.names = F)

#write.table(predicted_segment, file = paste0(basedir,"predicted_segment.txt"), sep=",", col.names = F,row.names = F, append = TRUE)
#write.table(posterior_probabilities, file = paste0(basedir,"posterior_probabilities.txt"), sep=",", col.names = F,row.names = F, append = TRUE)
#write.table(model_stats, file = paste0(basedir, "model_stats.txt"), sep=",", col.names = F,row.names = F, append = TRUE)

# now calculate ROV (= weighted chi-sq)
predicted_segment <- lapply(predicted_segment, as.character)
predicted_segment <- unlist(predicted_segment)
xtab_var_rows <- c(1:ncol(ds))

# ds_and_predicted_segment <- cbind(predicted_segment, ds)
weights <- read.csv(paste0(basedir,weights_filename))
weights <- lapply(weights, as.character)
weights <- unlist(weights)
# xtab_var_col <- c(1) # row where segment assignments are found
# xtab_var_rows <- c(2:ncol(ds)) # of var columns
rov_stats <- list()

#print (paste0('weights: ',weights))
#print (xtab_var_rows)

for (var in xtab_var_rows){
  #print (var)
  #print (ds[,var])  
  chisq_stat <- wtd.chi.sq(as.character(unlist(ds[,var])),predicted_segment,weight=as.numeric(weights),na.rm=TRUE)
  chisq_df <- ( length(unique(ds[,var])) - 1)  * ( length(unique(predicted_segment)) - 1) 
  chisq_stat <- chisq_stat[[1]] / 1
  rov_stat <- (chisq_stat - chisq_df) / sqrt(2*chisq_df)
  rov_stats[var] <- rov_stat
    }
# ncol(ds) == length(rov_stats)
rov_and_q <- cbind(colnames(ds),rov_stats)
write.table(rov_and_q, file = paste0(basedir,paste0(paste0("/static/model/rov_",timestamp),".txt")), sep=",", col.names = F,row.names = F)


