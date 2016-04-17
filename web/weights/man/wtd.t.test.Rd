\name{wtd.t.test}
\alias{wtd.t.test}
\title{
Produces weighted Student's t-tests with standard errors and significance.
}
\description{
\code{wtd.t.test} produces either one- or two-sample t-tests comparing weighted data streams to one another. Note that weights run with the default parameters here treat the weights as an estimate of the precision of the information. For survey data, users should run this code with bootstrapped standard errors \code{bootse=TRUE}, which are robust to heteroskadesticity, although these will vary slightly each time the weights are run. A prior version of this software was set to default to \code{mean1=FALSE} and \code{bootse=FALSE}.
}
\usage{
wtd.t.test(x, y=0, weight=NULL, weighty=NULL, samedata=TRUE,
alternative="two.tailed", mean1=TRUE, bootse=FALSE, bootp=FALSE,
bootn=1000, drops="pairwise")
}
\arguments{
  \item{x}{
\code{x} is a numerical vector which the researcher wishes to test against \code{y}.
}
  \item{y}{
\code{y} can be either a single number representing an alternative hypothesis or a second numerical vector which the researcher wishes to compare against \code{x}.
}
  \item{weight}{
\code{weight} is an optional vector of weights to be used to determine the weighted mean and variance for the \code{x} vector for all t-tests.  If \code{weighty} is unspecified and \code{samedata} is TRUE, this weight will be assumed to apply to both \code{x} and \code{y}.
}
  \item{weighty}{
\code{weighty} is an optional vector of weights to be used to determine the weighted mean and variance for the \code{y} vector for two-sample t-tests.  If \code{weighty} is unspecified and \code{samedata} is TRUE, this weight will be assumed to equal \code{weightx}.  If \code{weighty} is unspecified and \code{samedata} is FALSE, this weight will be assumed to equal 1 for all cases.
}
  \item{samedata}{
\code{samedata} is an optional identifier for whether the \code{x} and \code{y} data come from the same data stream for a two-sample test.  If true, \code{wtd.t.test} assumes that \code{weighty} should equal \code{weightx} if (1) \code{weighty} is unspecified, and (2) the lengths of the two vectors are identical.
}
  \item{alternative}{
\code{alternative} is an optional marker for whether one or two-tailed p-values shoould be returned.  By default, two-tailed values will be returned (\code{type="two.tailed"}).  To set to one-tailed values, alternative can be set to \code{type="greater"} to test \code{x>y} or \code{type="less"} to test \code{x<y}.
}
  \item{mean1}{
\code{mean1} is an optional parameter for determining whether the weights should be forced to have an average value of 1. If this is set as false, the weighted correlations will be produced with the assumption that the true N of the data is equivalent to the sum of the weights.
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
  \item{drops}{
\code{drops} is set to limit a t-test on the same data to cases with nonmissing data for x, y, and weights (if specified). If \code{drops} is anything other than \code{"pairwise"}, means for \code{x} and \code{y} are calculated on all available data rather than data that are available for both \code{x} and \code{y}. This parameter does nothing if \code{x} and \code{y} are not from the same dataset.
}
}
\value{
A list element with an identifier for the test; coefficients for the t value, degrees of freedom, and p value of the t-test; and additional statistics of potential interest.
}
\author{
Josh Pasek, Assistant Professor of Communication Studies at the University of Michigan (www.joshpasek.com).  Gene Culter added code for a one-tailed version of the test.
}
\seealso{
\code{\link{stdz}}
\code{\link{wtd.cor}}
\code{\link{wtd.chi.sq}}
}
\examples{
test <- c(1,1,1,1,1,1,2,2,2,3,3,3,4,4)
t2 <- rev(test)+1
weight <- c(.5,.5,.5,.5,.5,1,1,1,1,2,2,2,2,2)

wtd.t.test(test, t2)
wtd.t.test(test, t2, weight)
wtd.t.test(test, t2, weight, bootse=TRUE)
}
\keyword{ ~t.test }
\keyword{ ~weights }
\keyword{ ~bootstrap }
