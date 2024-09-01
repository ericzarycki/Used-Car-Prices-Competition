# Linear Regression in R

# Install packages

# install.packages("MASS")
# install.packages("ISLR2")
# install.packages("corrplot")
# install.packages("dplyr")
# install.packages("ggplot2")

# Load libraries

library(MASS)
library(ISLR2)
library(corrplot)
library(dplyr)
library(ggplot2)

# import datasets
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

#EDA
head(train)
str(train)
summary(train)

train.numeric <-
  train %>% select(where(is.numeric)) %>% select(!id)

#Distributions of numeric variables
hist(train.numeric$model_year)
hist(train.numeric$milage)
hist(train.numeric$price, breaks="FD", xlim=c(0,3000000))

train.cat <-
  train %>% select(where(is.character))
#Distributions of categorical variables
# to do 


train.matrix <- cor(train)


