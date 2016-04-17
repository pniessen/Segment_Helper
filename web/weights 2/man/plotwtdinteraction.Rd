\name{plotwtdinteraction}
\alias{plotwtdinteraction}
\alias{findwtdinteraction}
\alias{findwtdinteraction.default}
\alias{findwtdinteraction.mira}
\alias{plotinteractpreds}
\title{
Functions to Identify and Plot Predicted Probabilities As Well As Two- and Three-Way Interactions From Regressions With or Without Weights and Standard Errors
}
\description{
\code{plotwtdinteraction} produces a plot from a regression object to illustrate a two- or three-way interaction for a prototypical individual holding constant all other variables. Prototypical individual is identified as the mean (numeric), median (ordinal), and/or modal (factors and logical variables) values for all measures. Standard errors are illustrated with polygons by default.

\code{findwtdinteraction} generates a table of point estimates from a regression object to illustrate a two- or three-way interaction for a prototypical individual holding constant all other variables. Prototypical individual is identified as the mean (numeric), median (ordinal), and/or modal (factors and logical variables) values for all measures. Standard errors are illustrated with polygons by default.

\code{plotinteractpreds} plots an object from \code{findwtdinteraction}.

These functions are known to be compatible with \code{\link{lm}}, \code{\link{glm}}, as well as multiply imputed \code{\link{lm}} and \code{\link{glm}} data generated with the \code{\link{mice}} package.

*Note, this set of functions is still in beta, please let me know if you run into any bugs when using it.*

**Important: If you are using a regression output from a multiply imputed dataset with a continuous variable as an interacting term, you should always specify the levels (acrosslevs, bylevs, or atlevs) for the variable, as imputations can change the set of levels that are available and thus can make the point estimates across imputed datasets incompatible with one-another.**
}
\usage{
plotwtdinteraction(reg, across, by=NULL, at=NULL, acrosslevs=NULL, bylevs=NULL,
atlevs=NULL, weight=NULL, dvname=NULL, acclevnames=NULL, bylevnames=NULL,
atlevnames=NULL, stdzacross=FALSE, stdzby=FALSE, stdzat=FALSE, limitlevs=20,
type="response", seplot=TRUE, ylim=NULL, main=NULL, xlab=NULL, ylab=NULL,
legend=TRUE, placement="bottomright", lwd=3, add=FALSE, addat=FALSE, mfrow=NULL,
linecol=NULL, secol=NULL, showbynamelegend=FALSE, showatnamelegend=FALSE, ...)

findwtdinteraction(reg, across, by=NULL, at=NULL, acrosslevs=NULL, bylevs=NULL,
atlevs=NULL, weight=NULL, dvname=NULL, acclevnames=NULL, bylevnames=NULL,
atlevnames=NULL, stdzacross=FALSE, stdzby=FALSE, stdzat=FALSE, limitlevs=20,
type="response")

plotinteractpreds(out, seplot=TRUE, ylim=NULL, main=NULL, xlab=NULL, ylab=NULL,
legend=TRUE, placement="bottomright", lwd=3, add=FALSE, addat=FALSE, mfrow=NULL,
linecol=NULL, secol=NULL, showbynamelegend=FALSE, showatnamelegend=FALSE, ...)
}
\arguments{
\item{reg}{
\code{reg} is a regression object in lm, glm, or mira (multiply imputed) format that includes the variables to be plotted.
}
\item{out}{
\code{out} is an object estimate using findwtdinteraction that should be plotted.
}
\item{across}{
\code{across} specifies the name of the variable, in quotation marks, that was used in the regression that should be plotted on the X axis.
}
\item{by}{
\code{by} specifies the name of the variable, in quotation marks, that was used in the regression that should form each of the separate lines in the regression.
}
\item{at}{
\code{at} (optional) specifies the name of the variable, in quotation marks, that represents the third-way of a 3-way interaction.  Depending on specifications, this can either be plotted as additional lines or as separate graphs.
}
\item{acrosslevs}{
\code{acrosslevs} (optional) specifies the unique levels of the variable \code{across} that should be estimated across the x axis.  If this is not specified, each unique level of the \code{across} variable will be used.
}
\item{bylevs}{
\code{bylevs} (optional) specifies the unique levels of the variable \code{by} that should yield separate lines.  If this is not specified, each unique level of the \code{by} variable will be used.
}
\item{atlevs}{
\code{atlevs} (optional) specifies the unique levels of the variable \code{at} that should yield separate figures or lines. If this is not specified, each unique level of the \code{at} variable will be used.
}
\item{weight}{
\code{weight} (optional) allows the user to introduce a separate weight that was not used in the original regression.  If the regression was run using weights, those weights will always be used to generate estimates of the prototypical individual to be used.
}
\item{dvname}{
\code{dvname} (optional) allows the user to relabel the dependent variable for printouts.
}
\item{acclevnames}{
\code{dvname} (optional) allows the user to specify the names for the specified levels of the \code{accross} variable.
}
\item{bylevnames}{
\code{dvname} (optional) allows the user to specify the names for the specified levels of the \code{by} variable.
}
\item{atlevnames}{
\code{dvname} (optional) allows the user to specify the names for the specified levels of the \code{at} variable.
}
\item{stdzacross}{
\code{dvname} (optional) shows levels of \code{across} variable in (weighted) standard deviation units. This defaults to showing 1SD below mean and 1SD above mean; specifying \code{acrosslevs} to other values will provide results in SD units instead of variable units.
}
\item{stdzby}{
\code{dvname} (optional) shows levels of \code{by} variable in (weighted) standard deviation units. This defaults to showing 1SD below mean and 1SD above mean; specifying \code{bylevs} to other values will provide results in SD units instead of variable units. 
}
\item{stdzat}{
\code{dvname} (optional) shows levels of \code{at} variable in (weighted) standard deviation units. This defaults to showing 1SD below mean and 1SD above mean; specifying \code{atlevs} to other values will provide results in SD units instead of variable units. 
}
\item{limitlevs}{
\code{limitlevs} sets the number of different levels that any given interacting variable can have. This is meant to prevent inadvertent generation and plotting of tons of point estimates for continuous variables. The default is set to 20.
}
\item{type}{
\code{type} sets the type of prediction to be used for generation of the estimates. This defaults to \code{"response"} but can be used with any type of model prediction for which only one numeric estimate is given. (Not currently compatible with estimates derived from polr regression).
}
\item{seplot}{
\code{seplot} (optional) if set to \code{TRUE}, plots will include polygons illustrating standard errors.
}
\item{ylim}{
\code{ylim} (optional) passes on y-axis limits to \code{\link{plot}} function.
}
\item{main}{
\code{main} (optional) passes on title to \code{\link{plot}} function.
}
\item{xlab}{
\code{xlab} (optional) passes on x-axis labels to \code{\link{plot}} function.
}
\item{ylab}{
\code{ylab} (optional) passes on y-axis labels to \code{\link{plot}} function.
}
\item{legend}{
\code{legend} (optional) if \code{TRUE} will produce a legend on the interaction figure.
}
\item{placement}{
\code{placement} (optional) passes to \code{\link{legend}} function a location for the legend. Can be set to "bottomright", "bottomleft", "topright", and "topleft".
}
\item{lwd}{
\code{lwd} (optional) specifies the line strength for plots, this passes on to the \code{\link{plot}} command.
}
\item{add}{
\code{add} (optional) logical statement to add the results to an existing plot (\code{at=TRUE}) rather than generating a new one (\code{at=FALSE} is the default).
}
\item{addat}{
\code{addat} (optional) logical statement specifying whether the levels of \code{at} should be different plots (\code{addat=TRUE}) or if each level of \code{at} should generate a new plot (\code{addat=FALSE} is the default)
}
\item{mfrow}{
\code{mfrow} (optional) temporarily changes the number of plots per page in \code{\link{par}} for the purpose of generating current plots. This should generally only be used for 3-way interactions.  It takes commands of the form \code{c(2,3)}, specifying the number of rows and columns in the graphics interface. The algorithm defaults to putting all 3-way interactions on a single page with a width of 2. 
}
\item{linecol}{
\code{linecol} (optional) Specifies the colors of lines in the figure(s). For two-way interactions, this should be a vector of the same length as \code{bylevs}.  For 3-way interactions, the colors demarcate the levels of \code{at} instead and should be the same length as \code{atlevs}.
}
\item{secol}{
\code{secol} (optional) Specifies the colors of standard error in the figure(s). For two-way interactions, this should be a vector of the same length as \code{bylevs}.  For 3-way interactions, the colors demarcate the levels of \code{at} instead and should be the same length as \code{atlevs}.
}
\item{showbynamelegend}{
\code{showbynamelegend} (optional) adds name of \code{by} variable to names of value levels in legend.
}
\item{showatnamelegend}{
\code{showatnamelegend} (optional) adds name of \code{at} variable to names of value levels in legend.
}
\item{...}{
\code{...} (optional) Additional arguments to be passed on to plot command (or future methods of \code{findwtdinteraction}).
}
}
\value{
A table or figure illustrating the predicted values of the dependent variable across levels of the independent variables for a prototypical respondent.
}
\author{
Josh Pasek, Assistant Professor of Communication Studies at the University of Michigan (www.joshpasek.com).
}