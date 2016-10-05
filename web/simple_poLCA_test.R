# simple poLCA tester - to be run via Python wrapper
# 1.25.2016 Peter Niessen
#

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

print (vars)
print (weights_filename)


ds = read.csv(paste0(basedir,infile))

model_function <- as.formula(paste0("cbind(",toString(vars),")~1"))

print (model_function)


num_seg <- as.numeric(myArgs2[1])
print(num_seg)

nrep <- myArgs2[2]  


res2 = poLCA(model_function, maxiter=50000, nclass=num_seg, nrep=nrep, data=ds)


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

# now calculate ROV (= weighted chi-sq)
predicted_segment <- lapply(predicted_segment, as.character)
predicted_segment <- unlist(predicted_segment)
xtab_var_rows <- c(1:ncol(ds))

weights <- read.csv(paste0(basedir,weights_filename))
weights <- lapply(weights, as.character)
weights <- unlist(weights)

rov_stats <- list()

for (var in xtab_var_rows){

  chisq_stat <- wtd.chi.sq(as.character(unlist(ds[,var])),predicted_segment,weight=as.numeric(weights),na.rm=TRUE)
  print (chisq_stat)
  chisq_df <- ( length(unique(ds[,var])) - 1)  * ( length(unique(predicted_segment)) - 1) 
  chisq_stat <- chisq_stat[[1]] / 1
  rov_stat <- (chisq_stat - chisq_df) / sqrt(2*chisq_df)
  rov_stats[var] <- rov_stat
    }

rov_and_q <- cbind(colnames(ds),rov_stats)
write.table(rov_and_q, file = paste0(basedir,paste0(paste0("/static/model/rov_",timestamp),".txt")), sep=",", col.names = F,row.names = F)


