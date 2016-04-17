\name{wtd.cor}
\alias{wtd.cor}
\alias{onecor.wtd}
\title{
Produces weighted correlations with standard errors and significance.  For a faster version without standard errors and p values, use the \code{\link{wtd.cors}} function.
}
\description{
\code{wtd.cor} produces a Pearsons correlation coefficient comparing two variables or matrices.  Note that weights run with the default parameters here treat the weights as an estimate of the precision of the information.  For survey data, users should run this code with bootstrapped standard errors \code{bootse=TRUE}, which are robust to heteroskadesticity, although these will vary slightly each time the weights are run. A prior version of this software was set to default to \code{mean1=FALSE} and \code{bootse=FALSE}.
}
\usage{
wtd.cor(x, y=NULL, weight=NULL, mean1=TRUE, collapse=TRUE, bootse=FALSE,
bootp=FALSE, bootn=1000)
}
\arguments{
  \item{x}{
\code{x} should be a matrix or vector which the researcher wishes to correlate with \code{y}.
}
  \item{y}{
\code{y} should be a numerical vector or matrix which the researcher wishes to correlate with \code{x}.  If \code{y} is NULL, \code{x} will be used instead
}
  \item{weight}{
\code{weight} is an optional vector of weights to be used to determining the weighted mean and variance for calculation of the correlations.
}
  \item{mean1}{
\code{mean1} is an optional parameter for determining whether the weights should be forced to have an average value of 1. If this is set as false, the weighted correlations will be produced with the assumption that the true N of the data is equivalent to the sum of the weights.
}
  \item{collapse}{
\code{collapse} is an indicator for whether the data should be collapsed to a simpler form if either x or y is a vector instead of a matrix.
}
  \item{bootse}{
\code{bootse} is an optional parameter that produces bootstrapped standard errors. This should be used to address heteroskadesticity issues when weights indicate probabilities of selection rather than the precision of estimates.
}
  \item{bootp}{
\code{bootp} is an optional parameter that produces bootstrapped p values instead of estimating p values from the standard errors. This parameter only operates when \code{bootse=TRUE}.
}
  \item{bootn}{
\code{bootn} is an optional parameter that is used to indicate the number of bootstraps that should be run for \code{bootse} and \code{bootp}.
}
}
\value{
A list with matrices for the estimated correlation coefficient, the standard error on that correlation coefficient, the t-value for that correlation coefficient, and the p value for the significance of the correlation.  If the list can be simplified, simplification will be done.
}
\author{
Josh Pasek, Assistant Professor of Communication Studies at the University of Michigan (www.joshpasek.com).
}
\seealso{
\code{\link{wtd.cors}}
\code{\link{stdz}}
\code{\link{wtd.t.test}}
\code{\link{wtd.chi.sq}}
}
\examples{
test <- c(1,1,1,1,1,1,2,2,2,3,3,3,4,4)
t2 <- rev(test)
weight <- c(.5,.5,.5,.5,.5,1,1,1,1,2,2,2,2,2)

wtd.cor(test, t2)
wtd.cor(test, t2, weight)
wtd.cor(test, t2, weight, bootse=TRUE)
}
\keyword{ ~correlation }
\keyword{ ~Pearson }
\keyword{ ~weights }
\keyword{ ~bootstrap }
