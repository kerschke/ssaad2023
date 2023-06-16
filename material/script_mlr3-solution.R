library(tidyverse)
library(mlr3)
library(mlr3learners)
library(corrplot)

## 1. Import the detailed performance data.
x.perf = read_csv("data/algo_perf-detailed.csv")

## 2. Compute the ERT (sum of function evaluations divided by number of successful runs)
## for each combination of solver, dimension, function ID and precision.
x.ert = x.perf %>%
  group_by(solver, dim, fid, prec) %>%
  summarize(ERT = sum(fevals) / sum(succ)) %>%
  ungroup()

## 3. Visually compare the ERT performances (once for fixed dimension and once for fixed FID).
ggplot(data = filter(x.ert, dim == 2)) + 
  geom_line(mapping = aes(x = prec, y = ERT, color = solver)) +
  geom_point(mapping = aes(x = prec, y = ERT, color = solver)) +
  scale_y_log10() +
  scale_x_continuous(breaks = -7:1, trans = "reverse") +
  facet_wrap(~ fid, nrow = 6)

ggplot(data = filter(x.ert, fid == 22)) + 
  geom_line(mapping = aes(x = prec, y = ERT, color = solver)) +
  geom_point(mapping = aes(x = prec, y = ERT, color = solver)) +
  scale_y_log10() +
  scale_x_continuous(breaks = -7:1, trans = "reverse") +
  facet_wrap(~ dim, nrow = 1)

## 4. Compare the performance of BSqi and fmincon for fixed precision.
x.sample = x.ert %>%
  filter(solver %in% c("BSqi", "fmincon"), prec == -2) %>%
  pivot_wider(names_from = solver, values_from = ERT)

ggplot(x.sample) +
  geom_point(mapping = aes(x = BSqi, y = fmincon, shape = as.factor(dim), color = as.factor(dim))) +
  scale_x_log10() +
  scale_y_log10() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  guides(shape = guide_legend("Dimension"), color = guide_legend("Dimension"))


## 5. Import the feature data (values and costs).
feats = read_csv("data/feat_values.csv")
feat_costs = read_csv("data/feat_costs.csv")

## 6. Which feature sets are costly, and which ones are for free?
feat_costs %>%
  select(-(1:3)) %>%
  summarize_all(sum)

## 7. Remove all costly features from the feature set.
feats = feats %>%
  select(!starts_with(c("ela_conv", "ela_curv", "ela_local")))

## 8. Have a look at the correlations of the (remaining) features.
feats %>% 
  select(-(1:2)) %>% 
  cor(method = "spearman") %>% 
  corrplot::corrplot(method = "shade")

## 9. Remove some of the redundant features from the set.
feats.cleaned = feats %>%
  select(-c(
    "basic.cells_filled", "cm_angle.dist_ctr2best.mean", "cm_angle.dist_ctr2best.sd", 
    "cm_angle.dist_ctr2worst.mean", "cm_angle.dist_ctr2worst.sd", "cm_angle.angle.sd", 
    "cm_angle.y_ratio_best2worst.mean", "cm_angle.y_ratio_best2worst.sd", 
    "disp.ratio_mean_05", "disp.ratio_mean_10", "disp.ratio_mean_25", 
    "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10", 
    "disp.ratio_median_25", "disp.diff_mean_05", "disp.diff_mean_10", 
    "disp.diff_mean_25", "disp.diff_median_02", "disp.diff_median_05", 
    "disp.diff_median_10", "disp.diff_median_25"))

## 10. Have a closer look at the similarities of the remaining features.
feats.cleaned %>% 
  select(-(1:2)) %>% 
  cor(method = "spearman") %>% 
  corrplot::corrplot(method = "shade", order = "hclust")

## 11. Combine the performance and feature data into one joined data base and remove the "meta" data
df = left_join(x.sample, feats.cleaned, by = c("dim", "fid")) %>%
  select(-c("dim", "fid", "prec"))

## 12. Impute infinite ERT values by extremely large values (e.g., 1 mio)
impute.val = 1e7
df = df %>%
  mutate(
    BSqi = ifelse(is.finite(BSqi), BSqi, impute.val),
    fmincon = ifelse(is.finite(fmincon), fmincon, impute.val)
  )

## 13. Set up the corresponding classification task using mlr3 (https://mlr3book.mlr-org.com).
df.class = df %>%
  mutate(best = ifelse(BSqi < fmincon, "BSqi", ifelse(fmincon < BSqi, "fmincon", "both"))) %>%
  relocate(best, .before = "BSqi") %>%
  select(-c("BSqi", "fmincon"))

tsk = mlr3::as_task_classif(df.class, target = "best", id = "classif")


## 14. Train (and assess) a tree-based classifier.
tree = lrn("classif.rpart")
splits = partition(tsk, ratio = 0.8)
tree = tree$train(tsk, row_ids = splits$train)
pred.tree = tree$predict(tsk, row_ids = splits$test)
pred.tree$score()

# the code above performs a single split into train and test data, which
# might be insufficient; better use 10-fold crossvalidation
# for a more realistic estimate of the model performance
cv = rsmp("cv", folds = 10)
cv$instantiate(tsk)
r.tree = resample(tsk, tree, cv)
tree.acc = r.tree$score(msr("classif.ce"))
tree.acc[, .(iteration, classif.ce)]
r.tree$aggregate(msr("classif.ce"))

r.tree$prediction()$confusion

## 15. Do the same for a random forest.
ranger = lrn("classif.ranger")
ranger = ranger$train(tsk, row_ids = splits$train)
pred.ranger = ranger$predict(tsk, row_ids = splits$test)
pred.ranger$score()

cv = rsmp("cv", folds = 10)
cv$instantiate(tsk)
r.ranger = resample(tsk, tree, cv)
ranger.acc = r.ranger$score(msr("classif.ce"))
ranger.acc[, .(iteration, classif.ce)]
r.ranger$aggregate(msr("classif.ce"))

r.ranger$prediction()$confusion

## 16. Benchmark multiple algorithms against each other.
learners = lrns(c("classif.rpart", "classif.ranger", "classif.svm", "classif.featureless"), predict_type = "prob")
design = benchmark_grid(tsk, learners, cv)
bmr = benchmark(design)
bmr$score()
bmr$aggregate()[, .(task_id, learner_id, classif.ce)]
bmr$resample_results

bmr$resample_results$resample_result[[1]]$prediction()$confusion
bmr$resample_results$resample_result[[2]]$prediction()$confusion
bmr$resample_results$resample_result[[3]]$prediction()$confusion
bmr$resample_results$resample_result[[4]]$prediction()$confusion


## 17. Now, use a regression-based approach as algorithm selection model.

# in a first step, we need to produce two separate tasks and train separate models
df.bsqi = df %>% 
  select(-fmincon) %>%
  rename(ERT = BSqi)

tsk.bsqi = mlr3::as_task_regr(df.bsqi, target = "ERT", id = "BSqi")

learners.bsqi = lrns(c("regr.rpart", "regr.ranger", "regr.svm", "regr.lm", "regr.featureless"))
cv.bsqi = rsmp("cv", folds = 10)
cv.bsqi$instantiate(tsk.bsqi)
design.bsqi = benchmark_grid(tsk.bsqi, learners.bsqi, cv.bsqi)
bmr.bsqi = benchmark(design.bsqi)
bmr.bsqi$aggregate()[, .(task_id, learner_id, regr.mse)]

# extract predictions of ranger
pred.bsqi = bmr.bsqi$resample_results$resample_result[[2]]$prediction() %>%
  as.data.table() %>% 
  as_tibble() %>% 
  arrange(row_ids) %>%
  rename(response.BSqi = response, truth.BSqi = truth)

# do the same for the 2nd algorithm (here: fmincon)
df.fmincon = df %>% 
  select(-BSqi) %>%
  rename(ERT = fmincon)

tsk.fmin = mlr3::as_task_regr(df.fmincon, target = "ERT", id = "fmincon")

learners.fmin = lrns(c("regr.rpart", "regr.ranger", "regr.svm", "regr.lm", "regr.featureless"))
cv.fmin = rsmp("cv", folds = 10)
cv.fmin$instantiate(tsk.fmin)
design.fmin = benchmark_grid(tsk.fmin, learners.fmin, cv.fmin)
bmr.fmin = benchmark(design.fmin)
bmr.fmin$aggregate()[, .(task_id, learner_id, regr.mse)]

pred.fmin = bmr.fmin$resample_results$resample_result[[2]]$prediction() %>%
  as.data.table() %>% 
  as_tibble() %>% 
  arrange(row_ids) %>%
  rename(response.fmin = response, truth.fmin = truth)

# combine both predictions and identify, which algorithm should be selected
pred = left_join(pred.bsqi, pred.fmin, by = "row_ids") %>%
  mutate(
    truth = c("BSqi", "fmincon")[2 - (truth.BSqi < truth.fmin)],
    predicted = c("BSqi", "fmincon")[2 - (response.BSqi < response.fmin)]
  )

classif.ce = mean(pred$truth != pred$predicted)

# in my case, the classification approach (see 16.) resulted in a classification
# error of 0.2422 and the regression-based approach produced an error of 0.26 
# (note that these values could be different on your machine). So, in this case,
# the overhead of producing separate models per algorithm did not pay off.
