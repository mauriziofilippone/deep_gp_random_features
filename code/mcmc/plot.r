## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## Code to produce the plots in the paper - comparing MCMC samples with samples from the variational posterior in a two-layer DGP model with a Gaussian likelihood 

h = function(x) { exp(-x^2) * 2 * x }

## Set page size
ps.options(width=20, height=7.5, paper="special", horizontal=F, pointsize=36)
pdf.options(width=20, height=7.5, pointsize=36)

## Load data
X = as.matrix(read.table("X.txt"))
Xtest = as.matrix(read.table("Xtest.txt"))
Y = as.matrix(read.table("Y.txt"))

## Load predictions layer 1
predictions_MCMC = as.matrix(read.table("predictions_MCMC_F1.txt"))
predictions_var_NRFF_10 = as.matrix(read.table("predictions_variational_F1_NRFF_10.txt"))
predictions_var_NRFF_50 = as.matrix(read.table("predictions_variational_F1_NRFF_50.txt"))

nsamples = dim(predictions_MCMC)[2]

color_MCMC = rgb(1,0,0,alpha=0.02) 
color_var = rgb(0,0,1,alpha=0.02) 

## Open figure
pdf("figure_compare_MCMC_var.pdf")
par(mfrow=c(2,3),
    oma = c(1.1,2.1,1.5,0.2),
mar=c(0.3,0.5,0.0,0.0), mgp=c(1.1,0.1,0)
    ## mar = c(0,0,0,0) + 0.1
    )

## Produce plots layer 1

## par("mar"=c(1.5,1.5,0.3,0.3), "mgp"=c(1.8,0.6,0))
plot(Xtest, predictions_var_NRFF_10[,1], type="l", lwd=10, col=color_var, xlab="", ylab="Layer 1", ylim = c(-3,3), yaxt="n", xaxt="n") # , main="Variational - 10 RFF")
axis(2, at=c(-2, 0, 2), tck=0.02)
for(i in 2:nsamples) {
   points(Xtest, predictions_var_NRFF_10[,i], type="l", lwd=10, col=color_var)
}
points(Xtest, h(Xtest), type="l")

## par("mar"=c(1.5,1.5,0.3,0.3), "mgp"=c(1.8,0.6,0))
plot(Xtest, predictions_var_NRFF_50[,1], type="l", lwd=10, col=color_var, xlab="", ylab="", ylim = c(-3,3), yaxt="n", xaxt="n") # , main="Variational - 50 RFF")
for(i in 2:nsamples) {
   points(Xtest, predictions_var_NRFF_50[,i], type="l", lwd=10, col=color_var)
}
points(Xtest, h(Xtest), type="l")

## par("mar"=c(1.5,1.5,0.3,0.3), "mgp"=c(1.8,0.6,0))
plot(Xtest, predictions_MCMC[,1], type="l", lwd=10, col=color_MCMC, xlab="", ylab="", ylim = c(-3,3), yaxt="n", xaxt="n") #, main="MCMC")
for(i in 2:nsamples) {
   points(Xtest, predictions_MCMC[,i], type="l", lwd=10, col=color_MCMC)
}
points(Xtest, h(Xtest), type="l")

## ******************************

## Load predictions layer 2
predictions_MCMC = as.matrix(read.table("predictions_MCMC_F2.txt"))
predictions_var_NRFF_10 = as.matrix(read.table("predictions_variational_F2_NRFF_10.txt"))
predictions_var_NRFF_50 = as.matrix(read.table("predictions_variational_F2_NRFF_50.txt"))

## Produce plots layer 2

## par("mar"=c(1.5,1.5,0.3,0.3), "mgp"=c(1.8,0.6,0))
plot(Xtest, predictions_var_NRFF_10[,1], type="l", lwd=10, col=color_var, xlab="", ylab="Layer 2", ylim = c(-1,1), yaxt="n", tck=0.02) ##,  main="Layer 2 - Variational - 10 RFF", ylim = c(-1,1))
axis(2, at=c(-1, 0, 1), tck=0.02)
for(i in 2:nsamples) {
   points(Xtest, predictions_var_NRFF_10[,i], type="l", lwd=10, col=color_var)
}
points(X, Y, pch=20)
points(Xtest, h(h(Xtest)), type="l")

## par("mar"=c(1.5,1.5,0.3,0.3), "mgp"=c(1.8,0.6,0))
plot(Xtest, predictions_var_NRFF_50[,1], type="l", lwd=10, col=color_var, xlab="", ylab="", ylim = c(-1,1), yaxt="n", tck=0.02) ##, main="Layer 2 - Variational - 50 RFF", ylim = c(-1,1))
for(i in 2:nsamples) {
   points(Xtest, predictions_var_NRFF_50[,i], type="l", lwd=10, col=color_var)
}
points(X, Y, pch=20)
points(Xtest, h(h(Xtest)), type="l")

## par("mar"=c(1.5,1.5,0.3,0.3), "mgp"=c(1.8,0.6,0))
plot(Xtest, predictions_MCMC[,1], type="l", lwd=10, col=color_MCMC, xlab="", ylab="", ylim = c(-1,1), yaxt="n", tck=0.02) ##, main="Layer 2 - MCMC", ylim = c(-1,1))
for(i in 2:nsamples) {
   points(Xtest, predictions_MCMC[,i], type="l", lwd=10, col=color_MCMC)
}
points(X, Y, pch=20)
points(Xtest, h(h(Xtest)), type="l")

## ******************************

## Add legend text / titles
mtext('Variational - 10 RFF', side = 3, outer = TRUE, line = 0.3, at=0.18)
mtext('Variational - 50 RFF', side = 3, outer = TRUE, line = 0.3, at=0.51)
mtext('MCMC', side = 3, outer = TRUE, line = 0.3, at=0.85)
mtext('Layer 1', side = 2, outer = TRUE, line = 0.8, at=0.8)
mtext('Layer 2', side = 2, outer = TRUE, line = 0.8, at=0.25)


dev.off()
