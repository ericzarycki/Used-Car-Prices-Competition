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

# Load libraries

library(MASS)
library(ISLR2)
library(corrplot)
library(dplyr)
library(ggplot2)
library(caret)
library(xgboost)
library(fastDummies)
library(mltools)
library(data.table)
library(FeatureHashing)


# Import Train and Test Datasets from Kaggle
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")


##########################Exploratory Data Analysis###########################


#############Data Cleaning##################################################

#Check for missing values
colSums(is.na(train))
colSums(is.na(test))

##################################Feature Hashing for XGBoost################
#Remove and Store Predictor Variables in Train (ID and Price)
train_labels <- train$price
train_ids <- train$id
train.cleaned <- train %>% dplyr::select(-price, -id)
#And Test (ID)
test_ids <- test$id
test.cleaned <- test %>% dplyr::select(-id)

train.hashed <- hashed.model.matrix(~., data=train.cleaned, hash.size=2^6, transpose=FALSE)
test.hashed <- hashed.model.matrix(~., data = test.cleaned, hash.size = 2^6, transpose = FALSE)

dtrain <- xgb.DMatrix(train.hashed,label = train_labels)
dtest <- xgb.DMatrix(test.hashed)

mod.xgb <- xgboost(data = dtrain,
               label = train_labels,
               nrounds = 100,
               max_depth = 6,
               objective = 'reg:squarederror',
               verbose = 1)

#Feature Importance
# xgb_i <- xgb.importance(feature_names = colnames(train_data),
#                         model=bst)
# head(xgb_i)

# Predict on Test Set
predTest <- predict(mod.xgb, newdata=dtest)

#Measuring Model Performance
rmse <- sqrt(mean((train_labels - predTest)^2))
print(paste("RMSE:", rmse))

#Submission
submission <- data.frame(id = test$id, price = predTest)

# Save the submission file as CSV without row names
write.csv(submission, file = "submission.csv", row.names = FALSE)





#########################Creating the Model##############################
#Remove engine, transmission, fuel_type, model
# train.cleaned <- train %>% dplyr::select(-engine,-transmission,-model,-id, -ext_col,-int_col)
# 
# # Same for validation data
# test.cleaned <- test %>% dplyr::select(-engine,-transmission,-model,-id, -ext_col,-int_col)
# 
# ###########################One-hot encode############################
# 
# # One-hot encode categorical variables
# train.onehot <- dummy_cols(train.cleaned, select_columns = c("brand", "fuel_type",
#            "accident", "clean_title"))
# 
# 
# # One-hot encode categorical variables
# test.onehot <- dummy_cols(test.cleaned, select_columns = c("brand", "fuel_type", 
#                                                            "accident", "clean_title"))
# 
# ###############Remove Original Columns#########################
# train.onehot.cleaned <- 
#   train.onehot %>%
#   dplyr::select(-brand,-fuel_type,
#                 -accident, -clean_title, -brand_Polestar, -brand_smart)
# 
# # Remove original categorical columns (same as training)
# test.onehot.cleaned <- test.onehot %>% dplyr::select(-brand, -fuel_type, -accident, -clean_title)
# 
# #########################Make Sure everything is Numeric for Matrix#############
# 
# train.onehot.ready <- train.onehot.cleaned %>% mutate_if(is.integer, as.numeric)
# test.onehot.ready <- test.onehot.cleaned %>% mutate_if(is.integer, as.numeric)
# 
# 
# ############################MODEL############################
# set.seed(88)
# train_indices <- sample(1:nrow(train.onehot.ready), size = 0.8 * nrow(train.onehot.ready))
# 
# #
# train_data <- train.onehot.ready[train_indices, ]
# val_data <- train.onehot.ready[-train_indices, ]
# 
# train_labels <- train$price[train_indices]
# val_labels <- train$price[-train_indices]
# 
# train_data <- train_data %>% dplyr::select(-price)
# 
# dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_labels)
# dval <- xgb.DMatrix(data = as.matrix(val_data), label = val_labels)
# 
# # Model
# bst <- xgboost(data = as.matrix(train_data),
#                label = train_labels,
#                nrounds = 100,
#                max_depth = 6,
#                objective = 'reg:squarederror',
#                verbose = 1)
# 
# 
# 
# #Plot the Feature Importance for the Model
# xgb.plot.importance(xgb_i)



#####################################################################
#1 Submission RMSE: 86683.73
#2 Submission RMSE: 86096.4182781538

