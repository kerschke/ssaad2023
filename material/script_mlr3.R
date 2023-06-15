library(tidyverse)
library(mlr3)
library(mlr3learners)
library(corrplot)

## 1. Import the detailed performance data.
x.perf = read_csv("data/algo_perf-detailed.csv")

## 2. Compute the ERT (sum of function evaluations divided by number of successful runs)
## for each combination of solver, dimension, function ID and precision.
x.ert = ...

## 3. Visually compare the ERT performances (once for fixed dimension and once for fixed FID).
ggplot(data = filter(x.ert, dim == ...)) + 
  geom_line(mapping = aes(x = prec, y = ERT, color = solver)) +
  geom_point(mapping = aes(x = prec, y = ERT, color = solver)) +
  scale_y_log10() +
  scale_x_continuous(breaks = -7:1, trans = "reverse") +
  facet_wrap(~ fid, nrow = 6)

ggplot(data = filter(x.ert, fid == ...)) + 
  geom_line(mapping = aes(x = prec, y = ERT, color = solver)) +
  geom_point(mapping = aes(x = prec, y = ERT, color = solver)) +
  scale_y_log10() +
  scale_x_continuous(breaks = -7:1, trans = "reverse") +
  facet_wrap(~ dim, nrow = 1)

## 4. Pick two of the algorithms and compare their performances for a fixed precision.
x.sample = x.ert %>%
  filter(solver %in% c(...), prec == ...) %>%
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
## (Hint: An overview of flacco is available here: http://kerschke.github.io/flacco/)


## 7. Remove all costly features from the feature set.
feats = feats %>%
  select(!starts_with(c(...)))

## 8. Have a look at the correlations of the (remaining) features.
feats %>% 
  select(-(1:2)) %>% 
  cor(method = "spearman") %>% 
  corrplot::corrplot(method = "shade")

## 9. Remove some of the redundant features from the set.

## 10. Have a closer look at the similarities of the remaining features.


## 11. Combine the performance and feature data into one joined data base and remove the "meta" data
df = ...

## 12. Impute infinite ERT values by extremely large values (e.g., 1 mio)
impute.val = 1e7
df = df %>%
  mutate(
    ...
  )

## 13. Set up the corresponding classification task (see https://mlr3book.mlr-org.com).
df.class = df %>%
  mutate(best = ...)

tsk = mlr3::as_task_classif(df.class, target = "best", id = "classif")


## 14. Train (and assess) a tree-based classifier.
tree = lrn("classif.rpart")
tree = tree$train(tsk)
pred.tree = tree$predict(tsk)
pred.tree$score()

## more plausible results using 10-fold crossvalidation
cv = rsmp("cv", folds = 10)
cv$instantiate(tsk)
r.tree = resample(tsk, tree, cv)
tree.acc = r.tree$score(msr("classif.ce"))
tree.acc[, .(iteration, classif.ce)]
r.tree$aggregate(msr("classif.ce"))

r.tree$prediction()$confusion

## 15. Do the same for a random forest.


## 16. Benchmark multiple algorithms against each other.
learners = lrns(c("classif.rpart", "classif.ranger", "classif.svm", "classif.featureless"), predict_type = "prob")
design = benchmark_grid(tsk, learners, cv)
bmr = benchmark(design)
bmr$score()
bmr$aggregate()[, .(task_id, learner_id, classif.ce)]
bmr$resample_results

## assess the confusion matrices of each ML algorithm
bmr$resample_results$resample_result[[1]]$prediction()$confusion
bmr$resample_results$resample_result[[2]]$prediction()$confusion
bmr$resample_results$resample_result[[3]]$prediction()$confusion
bmr$resample_results$resample_result[[4]]$prediction()$confusion


## 17. Now, use a regression-based approach as algorithm selection model.
