# Preregistered fallback / cross-check of the measurement-error model in MCMCglmm (prereg sec. 6).
# Known measurement variance m^2 enters via mev; phylogenetic (A from MCC) + non-phylogenetic species random effects.
# Usage: Rscript me_mcmcglmm.R <me_data.csv> <mcc.tre> <out_dir> <m_column>
args <- commandArgs(TRUE)
.libPaths("C:/Users/Daglar/tools/R-lib")
suppressPackageStartupMessages({library(ape); library(MCMCglmm)})
d <- read.csv(args[1]); mcol <- args[4]; out <- args[3]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
d$phylo <- d$species; d$sp <- d$species
tr <- keep.tip(read.tree(args[2]), unique(d$species))
tr$node.label <- NULL
Ainv <- inverseA(tr, nodes = "TIPS", scale = TRUE)$Ainv
pr <- list(B = list(mu = rep(0, 6), V = diag(c(2.5^2, rep(0.5^2, 5)))),
           R = list(V = 1, nu = 0.002), G = list(G1 = list(V = 1, nu = 1, alpha.mu = 0, alpha.V = 0.25),
                                                  G2 = list(V = 1, nu = 1, alpha.mu = 0, alpha.V = 0.25)))
res <- list()
for (ch in 1:4) {
  set.seed(1000 + ch)
  res[[ch]] <- MCMCglmm(log10_brain ~ log10_body + log10_colony + iodine + vox122 + vox611, random = ~phylo + sp,
                        ginverse = list(phylo = Ainv), mev = d[[mcol]]^2, data = d, prior = pr,
                        nitt = 260000, burnin = 10000, thin = 100, verbose = FALSE, pr = FALSE)
}
library(coda)
sol <- mcmc.list(lapply(res, function(r) r$Sol[, 1:6])); vc <- mcmc.list(lapply(res, function(r) r$VCV))
b <- as.numeric(unlist(lapply(res, function(r) r$Sol[, "log10_colony"])))
g <- gelman.diag(sol, multivariate = FALSE)$psrf[, 1]; gv <- gelman.diag(vc, multivariate = FALSE)$psrf[, 1]
cat(sprintf("colony: mean=%.4f median=%.4f 95%%CrI=[%.4f, %.4f] P(>0)=%.4f\n", mean(b), median(b), quantile(b, .025), quantile(b, .975), mean(b > 0)))
cat("ESS fixed:", round(effectiveSize(sol)), "\nESS VCV:", round(effectiveSize(vc)), "\nPSRF fixed:", round(g, 4), "\nPSRF VCV:", round(gv, 4), "\n")
s <- summary(sol)$statistics; q <- summary(sol)$quantiles
write.csv(data.frame(term = rownames(s), mean = s[, 1], q025 = q[, 1], q975 = q[, 5], ess = effectiveSize(sol), psrf = g),
          file.path(out, paste0("mcmcglmm_summary_", mcol, ".csv")), row.names = FALSE)
