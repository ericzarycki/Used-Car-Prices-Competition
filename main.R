# Install packages

# install.packages("MASS")
# install.packages("ISLR2")
# install.packages("corrplot")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("caret")
# install.packages("xgboost")
# install.packages("mltools")
# install.packages("data.table")
# install.packages("FeatureHashing")
# install.packages("fastDummies") # For one-hot
# install.packages("tidymodels")

# Load libraries

library(MASS)
library(ISLR2)
library(corrplot)
library(dplyr)
library(ggplot2)
library(caret)
library(mltools)
library(data.table)
library(tidymodels)


# Import Train and Test Datasets from Kaggle
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

train.df <- train %>% select(price, brand, model, engine, fuel_type, transmission, accident, model_year, milage, ext_col, int_col) %>%
  mutate(price = log10(price)) %>% mutate_if(is.character, factor)
test.df <- test %>% select(brand, model, engine, fuel_type, transmission, accident, model_year, milage, ext_col, int_col)

#
set.seed(88)
train.split <- initial_split(train.df, strata = price)
train.t <- training(train.split)
train.v <- testing(train.split)

#Re-sampl
train.fold <- vfold_cv(train.t, strata = price, v=5)
train.fold

library(usemodels)

use_ranger(price ~., data = train.t)

#######
library(textrecipes)
library(hardhat)
ranger_recipe <- 
  recipe(formula = price ~ ., data = train.t) %>%
  step_other(brand,model,engine,transmission, ext_col,int_col, threshold = 0.01) %>% textrecipes::step_clean_levels(model,engine,transmission,ext_col, int_col)


ranger_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 10) %>% 
  set_mode("regression") %>% 
  set_engine("ranger") 

ranger_workflow <- 
  workflow() %>% 
  add_recipe(ranger_recipe,
             blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)) %>% 
  add_model(ranger_spec) 

set.seed(11928)
doParallel::registerDoParallel()
ranger_tune <-
  tune_grid(ranger_workflow, resamples = train.fold, grid = 11)


show_best(ranger_tune, metric="rmse")
show_best(ranger_tune, metric="rsq")

autoplot(ranger_tune)


#Feature Importance
library(vip)
imp_spec <- ranger_spec %>% finalize_model(select_best(ranger_tune)) %>% set_engine("ranger", importance="permutation")
workflow() %>% add_recipe(ranger_recipe) %>% add_model(imp_spec) %>% fit(train.t) %>% vip()

final_rf <-
  ranger_workflow %>% finalize_workflow(select_best(ranger_tune))

train.fit <- last_fit(final_rf, train.split)

collect_metrics(train.fit)

collect_predictions(train.fit) %>% ggplot(aes(price,.pred))+
  geom_abline(lty=2, color="gray50") +
  geom_point(alpha = 0.5, color="midnightblue") +
  coord_fixed()

########################SUBMISSION
final_rf_fitted <- fit(final_rf, data = train.df)
test.pred <- predict(final_rf_fitted, new_data = test.df)
submission <- data.frame(id = test$id, price = 10^test.pred$.pred)
write.csv(submission, "submission.csv", row.names = FALSE)
