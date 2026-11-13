# Cross-check of src/analysis/pgls.py against R: Pagel-lambda PGLS with phylolm and nlme::gls(corPagel).
# Usage: Rscript phylolm_check.R <tree.tre> <match.csv: species_key,tip> <data.csv: species_key,...> <formula> <out.csv>
suppressPackageStartupMessages({library(ape); library(phylolm); library(nlme)})
a <- commandArgs(trailingOnly = TRUE)
tree <- read.tree(a[1]); m <- read.csv(a[2], stringsAsFactors = FALSE); d <- read.csv(a[3], stringsAsFactors = FALSE)
f <- as.formula(a[4])
m <- m[!is.na(m$tip) & m$species_key %in% d$species_key, ]
tree <- keep.tip(tree, m$tip)
tree$tip.label <- m$species_key[match(tree$tip.label, m$tip)]
d <- d[match(tree$tip.label, d$species_key), ]; rownames(d) <- d$species_key
fit <- phylolm(f, data = d, phy = tree, model = "lambda")
s <- summary(fit)$coefficients
o1 <- data.frame(engine = "phylolm", term = rownames(s), estimate = s[, 1], se = s[, 2], p = s[, 4],
                 lambda = fit$optpar, logLik = fit$logLik, n = nrow(d))
g <- gls(f, data = d, correlation = corPagel(0.5, tree, form = ~species_key), method = "ML")
t <- summary(g)$tTable
o2 <- data.frame(engine = "nlme_corPagel", term = rownames(t), estimate = t[, 1], se = t[, 2], p = t[, 4],
                 lambda = as.numeric(coef(g$modelStruct$corStruct, unconstrained = FALSE)), logLik = as.numeric(logLik(g)), n = nrow(d))
o <- rbind(o1, o2); rownames(o) <- NULL
write.csv(o, a[5], row.names = FALSE); print(o)
