# Preregistered measurement-error model (prereg_draft.md sec. 6), specimen level, brms.
#   log10 Brain_ij | se(m_ij, sigma = TRUE) ~ log10 Body_ij + log10 Colony_j + Iodine_ij + VoxClass_ij
#                                            + (1 | gr(species, cov = A)) + (1 | species)
# Priors: b ~ N(0, 0.5); Intercept ~ t3(0, 2.5); all SDs (incl. sigma) ~ half-t3(0, 0.5). 4 chains x 2000 iterations.
# Usage: Rscript me_brms.R <me_data.csv> <mcc.tre> <out_dir> <m_column> [backend] [iter]
args <- commandArgs(TRUE)
.libPaths("C:/Users/Daglar/tools/R-lib")
Sys.setenv(RTOOLS45_HOME = "C:/Users/Daglar/tools/rtools45")
Sys.setenv(PATH = paste("C:/Users/Daglar/tools/rtools45/usr/bin", "C:/Users/Daglar/tools/rtools45/x86_64-w64-mingw32.static.posix/bin", Sys.getenv("PATH"), sep = ";"))
suppressPackageStartupMessages({library(ape); library(brms); library(posterior)})
backend <- if (length(args) >= 5) args[5] else "cmdstanr"
niter <- if (length(args) >= 6) as.integer(args[6]) else 2000L
if (backend == "cmdstanr") cmdstanr::set_cmdstan_path(list.dirs("C:/Users/Daglar/tools/cmdstan", recursive = FALSE)[1])
d <- read.csv(args[1]); mcol <- args[4]; out <- args[3]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
d$m <- d[[mcol]]; d$phylo <- d$species
tr <- read.tree(args[2]); tr <- keep.tip(tr, unique(d$species))
A <- vcv(tr, corr = TRUE)
pri <- c(prior(normal(0, 0.5), class = "b"), prior(student_t(3, 0, 2.5), class = "Intercept"),
         prior(student_t(3, 0, 0.5), class = "sd"), prior(student_t(3, 0, 0.5), class = "sigma"))
f <- bf(log10_brain | se(m, sigma = TRUE) ~ log10_body + log10_colony + iodine + vox122 + vox611 +
          (1 | gr(phylo, cov = A)) + (1 | species))
t0 <- Sys.time()
fit <- brm(f, data = d, data2 = list(A = A), prior = pri, chains = 4, cores = 4, iter = niter, warmup = 1000,
           seed = 20261115, control = list(adapt_delta = 0.95, max_treedepth = 12), backend = backend, refresh = 0)
dt <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
s <- summarise_draws(as_draws_df(fit), "mean", "median", "sd", ~quantile(.x, c(0.025, 0.05, 0.95, 0.975)), "rhat", "ess_bulk", "ess_tail")
s <- s[!grepl("^r_|^lp__|^lprior|^z_", s$variable), ]
write.csv(s, file.path(out, paste0("brms_summary_", mcol, "_iter", niter, ".csv")), row.names = FALSE)
all <- summarise_draws(as_draws_df(fit), "rhat", "ess_bulk", "ess_tail")
np_ <- nuts_params(fit); nd <- sum(np_$Value[np_$Parameter == "divergent__"])
cat(sprintf("n_obs=%d n_species=%d minutes=%.1f max_rhat_all=%.4f min_ess_bulk_all=%.0f min_ess_tail_all=%.0f divergent=%d\n",
            nrow(d), length(unique(d$species)), dt, max(all$rhat, na.rm = TRUE), min(all$ess_bulk, na.rm = TRUE),
            min(all$ess_tail, na.rm = TRUE), nd))
print(as.data.frame(s), digits = 4)
pc <- as_draws_df(fit)$b_log10_colony
cat(sprintf("P(b_colony > 0) = %.4f; P(b_colony > 0.02) = %.4f\n", mean(pc > 0), mean(pc > 0.02)))
saveRDS(fit, file.path(out, paste0("brms_fit_", mcol, "_iter", niter, ".rds")))
