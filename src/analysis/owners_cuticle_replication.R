# Replicate Matte et al. 2025 (Sci Adv) Table S2, model 2, with the owners' own R functions.
# Functions below are copied verbatim from R_scripts.zip (resources/cuticle_00_personal_library.R, load_data.R,
# load_trees.R, load_trait_data.R); only file paths are changed. The owners' private name-update table
# (phylogeny_names.txt) is not public, so update_phylo_names() is the identity here.
# Usage: Rscript owners_cuticle_replication.R <datasets_dir> <trees_dir> <out_dir>
suppressPackageStartupMessages({library(MASS); library(nlme); library(ape); library(dplyr); library(readxl)})
args <- commandArgs(trailingOnly = TRUE)
dd <- args[1]; td <- args[2]; od <- args[3]; dir.create(od, showWarnings = FALSE, recursive = TRUE)
`%notin%` <- Negate(`%in%`)

## ---- verbatim owners' functions ----
expend_dataset <- function(datasetA, datasetB, column_name) {
  datasetB <- subset(datasetB, get(column_name) %in% datasetA[[column_name]])
  lacking_row <- setdiff(datasetA[[column_name]], datasetB[[column_name]])
  if (length(lacking_row)>0) {
    new_rows <- data.frame(temp = lacking_row)
    colnames(new_rows) <- column_name
    for (i in setdiff(colnames(datasetB), column_name)){new_rows[[i]] <- NA}
    datasetB <- bind_rows(datasetB, new_rows)
  }
  return(datasetB)
}
place_lacking_species <- function(sp_list, phy) {
  name_to_assign <- setdiff(sp_list, phy$tip.label)
  unused_branch <- setdiff(phy$tip.label, sp_list)
  used_branch <- phy$tip.label[phy$tip.label %in% sp_list]
  unused_branch_genus <- sub("\\..*", "", unused_branch)
  used_branch_genus <- sub("\\..*", "", used_branch)
  tips_to_change = c(); new_tips_name = c()
  for (name in name_to_assign) {
    genus <-  sub("\\..*", "", name)
    if (genus %in% unused_branch_genus && name %notin% phy$tip.label) {
      potential_tips <- unused_branch[which(unused_branch_genus == genus)]
      if (length(potential_tips) > 0) {
        potential_tip_index <- which(phy$tip.label %in% potential_tips)
        already_existing_index <- which(phy$tip.label %in% used_branch[used_branch_genus==genus])
        if (length(already_existing_index) > 0) {
          distances <- sapply(potential_tip_index, function(x) min(abs(x - already_existing_index)))
          selected_index <- potential_tip_index[which.min(distances)]
        }else{selected_index <- potential_tip_index[1]}
        old_tips_name <- phy$tip.label[selected_index]
        idx_to_del <- which(unused_branch == old_tips_name)
        unused_branch <- unused_branch[-idx_to_del]
        unused_branch_genus <- unused_branch_genus[-idx_to_del]
        tips_to_change = c(tips_to_change, old_tips_name)
        new_tips_name = c(new_tips_name, name)
        phy$tip.label[selected_index] <- name
      }
    }
  }
  list(phy = phy, tips_to_change = tips_to_change, new_tips_name = new_tips_name)
}
assigne_tip_to_lacking_species <- function(phy, tips_to_change, new_tips_name) {
  phy$tip.label[which(phy$tip.label %in% tips_to_change )] <- new_tips_name
  return(phy)
}
# corrected variant (ours): rename each old tip to ITS new name, independent of tip order
assign_tip_fixed <- function(phy, tips_to_change, new_tips_name) {
  i <- match(tips_to_change, phy$tip.label); phy$tip.label[i] <- new_tips_name; phy
}
compute_mean_vcv <- function (multiphylo) {
  for (i in (1: length(multiphylo))) {
    vcv <- vcv.phylo(as.phylo(multiphylo[[i]]))
    if (i == 1) {mean_vcv <- vcv
    } else {mean_vcv <- mean_vcv + vcv[match(rownames(mean_vcv),rownames(vcv)),match(colnames(mean_vcv),colnames(vcv))]}
  }
  mean_vcv / length(multiphylo)
}
process_data_for_PGLS <- function(formula, dataset, cov_mat) {
  dataset_pruned <- dataset[complete.cases(dataset[, all.vars(formula)]), ]
  cov_mat_pruned <- cov_mat[rownames(cov_mat) %in% dataset_pruned$genus.species, colnames(cov_mat) %in% dataset_pruned$genus.species]
  dataset_pruned <- dataset_pruned[order(match(dataset_pruned$genus.species, rownames(cov_mat_pruned))), , drop = FALSE]
  list(dataset_pruned = dataset_pruned, cov_mat_pruned = cov_mat_pruned)
}
mean_per_species <- function(dataset) {
  dataset %>% dplyr::select(-vial_label, -subcaste , -volex_size) %>% group_by(genus.species) %>%
    summarise(subfamily = first(subfamily), genus = first(genus), caste = first(caste),
              body_volume = mean(body_volume, na.rm = TRUE), cuticle_volume = mean(cuticle_volume, na.rm = TRUE),
              cuticle_investment = mean(cuticle_investment, na.rm = TRUE)) %>% distinct()
}

## ---- load_data.R ----
dataset <- as.data.frame(readxl::read_excel(file.path(dd, "Dataset_S1_individual_measurements.xlsx")))
data_w <- subset(dataset, caste == "worker")
data_w_mean <- as.data.frame(mean_per_species(data_w))
cat("workers:", nrow(data_w), " species:", nrow(data_w_mean), "\n")

## ---- load_trees.R (name update = identity) ----
temp <- data_w_mean
tree_mcc <- read.tree(file.path(td, "15k_NCuniform_stem_mcc.tre"))
res_place <- place_lacking_species(temp$genus.species, tree_mcc)
tree_mcc <- res_place$phy
no_phylo <- setdiff(temp$genus.species, tree_mcc$tip.label)
cat("in tree directly:", sum(data_w_mean$genus.species %in% read.tree(file.path(td, "15k_NCuniform_stem_mcc.tre"))$tip.label),
    " placed on same-genus tip:", length(res_place$new_tips_name), " no phylo:", length(no_phylo), "\n")
writeLines(no_phylo, file.path(od, "owners_no_phylo.txt"))
write.csv(data.frame(old_tip = res_place$tips_to_change, new_name = res_place$new_tips_name), file.path(od, "owners_placed.csv"), row.names = FALSE)
temp <- subset(temp, temp$genus.species %in% tree_mcc$tip.label)

two <- c(read.tree(file.path(td, "15K_FBD_crown_posterior.tre")), read.tree(file.path(td, "15K_FBD_stem_posterior.tre")))
cat("posterior trees:", length(two), "\n")
prune_set <- function(assign_fun) {
  lapply(two, function(p) { p <- assign_fun(p, res_place$tips_to_change, res_place$new_tips_name)
                            ape::drop.tip(p, setdiff(p$tip.label, temp$genus.species)) })
}
two_bug <- prune_set(assigne_tip_to_lacking_species)
two_fix <- prune_set(assign_tip_fixed)
# how often does the owners' positional assignment put a species on a tip of ANOTHER genus?
mis <- sapply(two, function(p) { i <- which(p$tip.label %in% res_place$tips_to_change)
  sum(sub("\\..*", "", p$tip.label[i]) != sub("\\..*", "", res_place$new_tips_name)) })
cat("owners' assign: species put on a tip of another genus, per tree: median", median(mis), "range", range(mis), "of", length(res_place$new_tips_name), "\n")
heights <- sapply(two_fix, function(p) max(node.depth.edgelength(p)))
cat("tree heights (pruned): crown median", median(heights[1:100]), " stem median", median(heights[101:200]), "\n")
writeLines(sprintf('{"worker_species": %d, "in_tree_exact": %d, "genus_placed": %d, "no_phylo": %d, "wrong_genus_per_tree_median": %g, "wrong_genus_per_tree_min": %d, "wrong_genus_per_tree_max": %d, "tree_height_median": %g, "R": "%s", "nlme": "%s", "ape": "%s"}',
  nrow(data_w_mean), sum(data_w_mean$genus.species %in% read.tree(file.path(td, "15k_NCuniform_stem_mcc.tre"))$tip.label),
  length(res_place$new_tips_name), length(no_phylo), median(mis), min(mis), max(mis), median(heights),
  R.version.string, as.character(packageVersion("nlme")), as.character(packageVersion("ape"))), file.path(od, "owners_diagnostics.json"))

## ---- load_trait_data.R ----
add_traits <- function(dataset) {
  cs <- readxl::read_excel(file.path(dd, "Dataset_S3_colony_size.xlsx")) %>% as.data.frame() %>%
    rename(genus.species = `Valid Species Name`) %>% group_by(genus.species) %>%
    summarise(genus = first(`Valid Genus`), colony_size = median(`Mean Size`, na.rm = TRUE)) %>%
    group_by(genus) %>% summarise(colony_size = median(colony_size, na.rm = TRUE))
  cs <- expend_dataset(dataset, cs, 'genus'); dataset <- inner_join(dataset, cs, by = "genus")
  sp <- readxl::read_excel(file.path(dd, "Dataset_S4_cuticular_spines.xlsx")) %>% dplyr::select(genus, spine_number)
  sp <- expend_dataset(dataset, sp, 'genus'); dataset <- inner_join(dataset, sp, by = "genus")
  cl <- readxl::read_excel(file.path(dd, "Dataset_S5_climatic_data.xlsx"))
  cl <- expend_dataset(dataset, cl, 'genus.species'); dataset <- inner_join(dataset, cl, by = "genus.species", multiple = "all")
  di <- readxl::read_excel(file.path(dd, "Dataset_S6_species_diets.xlsx")) %>% dplyr::select(genus.species, diet)
  di <- expend_dataset(dataset, di, 'genus.species'); dataset <- inner_join(dataset, di, by = "genus.species")
  es <- readxl::read_excel(file.path(dd, "Dataset_S7_ecological_stratification.xlsx")) %>%
    mutate(foraging = ifelse(spe_arboreal == 1, 'arboreal', ifelse(spe_hypogaeic == 1, 'hypogaeic', 'epigaeic'))) %>% dplyr::select(genus, foraging)
  es <- expend_dataset(dataset, es, 'genus'); dataset <- inner_join(dataset, es, by = "genus")
  dataset
}

## ---- cuticle_02_PGLS_analysis.R, model 1 and model 2 ----
f1 <- log10(cuticle_volume) ~ mean_ann_prec + mean_ann_temp + colony_size + spine_number + foraging + diet + log10(body_volume)
f2 <- log10(cuticle_volume) ~ mean_ann_temp + colony_size + spine_number + diet + log10(body_volume)
run_one <- function(two_pruned, tag) {
  cov_mat <- compute_mean_vcv(two_pruned) / 200
  ds <- add_traits(temp)
  ds$diet <- factor(ds$diet, levels = c('omnivore', 'predator', 'herbivore', "fungivore"))
  ds$foraging <- factor(ds$foraging, levels = c('epigaeic', 'arboreal', 'hypogaeic'))
  ds$log10_colony_raw <- log10(ds$colony_size)
  ds$mean_ann_prec <- as.numeric(scale(ds$mean_ann_prec)); ds$mean_ann_temp <- as.numeric(scale(ds$mean_ann_temp))
  ds$colony_size <- as.numeric(scale(log10(ds$colony_size))); ds$spine_number <- as.numeric(scale(ds$spine_number))
  out <- list()
  for (m in list(list("model1", f1), list("model2", f2))) {
    r <- process_data_for_PGLS(m[[2]], ds, cov_mat)
    fit <- gls(m[[2]], correlation = corSymm(r$cov_mat_pruned[lower.tri(r$cov_mat_pruned)], fixed = TRUE), method = "ML", data = r$dataset_pruned)
    tt <- as.data.frame(summary(fit)$tTable); tt$term <- rownames(tt); tt$model <- m[[1]]; tt$variant <- tag
    tt$n <- fit$dims$N; tt$df <- fit$dims$N - fit$dims$p; tt$logLik <- as.numeric(logLik(fit))
    tt$sd_log10_colony <- sd(r$dataset_pruned$log10_colony_raw)
    out[[m[[1]]]] <- tt
    if (m[[1]] == "model2") {
      write.csv(r$dataset_pruned, file.path(od, paste0("model2_data_", tag, ".csv")), row.names = FALSE)
      write.csv(r$cov_mat_pruned, file.path(od, paste0("model2_cov_", tag, ".csv")))
    }
  }
  do.call(rbind, out)
}
res <- rbind(run_one(two_bug, "owners_code"), run_one(two_fix, "fixed_tip_assignment"))
rownames(res) <- NULL
write.csv(res, file.path(od, "owners_replication_table.csv"), row.names = FALSE)
print(res[res$term %in% c("colony_size", "log10(body_volume)"), c("variant","model","term","Value","Std.Error","t-value","p-value","n","df","sd_log10_colony")], digits = 6)
