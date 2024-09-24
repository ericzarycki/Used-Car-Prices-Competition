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


# import datasets
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

#Check for missing values
colSums(is.na(train))
colSums(is.na(test))

#Exploratory Data Analysis (TBD)
head(train)

summary(train)

# How many distinct variables are there?

train %>% distinct()


#Distributions of numeric variables
# hist(train.numeric$model_year)
# hist(train.numeric$milage)
# hist(train.numeric$price, breaks="FD", xlim=c(0,3000000))
# 
# train.cat <-
#   train %>% select(where(is.character))
#Distributions of categorical variables
# to do 
# ggplot(train.cat, aes(y=reorder(brand, sum) sum)) + geom_bar()

# Correlation matrix tbd
# train.matrix <- cor(train)

#########################Creating the Model##############################
#Remove engine, transmission, fuel_type, model
train.cleaned <- train %>% select(-engine,-transmission,-model,-id)

# Pre-processing 

#Turn integers into numeric values
train.numeric <- train.cleaned %>% mutate_if(is.integer, as.numeric)
str(train.numeric)

# One-hot encode categorical variables
train.onehot <- dummy_cols(train.numeric, select_columns = c("brand", "fuel_type", "ext_col", "int_col",
           "accident", "clean_title"))

#Remove col names
train.onehot.cleaned <- 
  train.onehot %>%
    select(-brand,-fuel_type,-ext_col, -int_col,
           -accident, -clean_title)

# make everything numeric because

train.onehot.ready <- train.onehot.cleaned %>% mutate_if(is.integer, as.numeric)
# Feature Importance

#split this into training and testing sets
set.seed(88)
train_indices <- sample(1:nrow(train.onehot.ready), size = 0.8 * nrow(train.onehot.ready))

#
train_data <- train.onehot.ready[train_indices, ]
val_data <- train.onehot.ready[-train_indices, ]

train_labels <- train$price[train_indices]
val_labels <- train$price[-train_indices]

# Model
bst <- xgboost(data = as.matrix(train_data),
               label = train_labels,
               nrounds = 100,
               max_depth = 6,
               objective = 'reg:squarederror',
               verbose = 1)




# Assuming you've done similar preprocessing for the test set:
test.cleaned <- test %>% select(-engine, -transmission, -model, -id)

# Turn integers into numeric values
test.numeric <- test.cleaned %>% mutate_if(is.integer, as.numeric)

# One-hot encode categorical variables
test.onehot <- dummy_cols(test.numeric, select_columns = c("brand", "fuel_type", "ext_col", "int_col",
                                                           "accident", "clean_title"))

# Remove original categorical columns (same as training)
test.onehot.cleaned <- test.onehot %>% select(-brand, -fuel_type, -ext_col, -int_col, -accident, -clean_title)

# Ensure all data is numeric
test.onehot.ready <- test.onehot.cleaned %>% mutate_if(is.integer, as.numeric)


# Get the feature names from the test set
test_features <- colnames(test.onehot.ready)

# Identify missing columns in the test set
missing_cols <- setdiff(train_features, test_features)





# Assuming your test set still contains the 'id' column
submission <- data.frame(id = test$id, price = predictions)

# Save the submission file as CSV without row names
write.csv(submission, file = "submission.csv", row.names = FALSE)

