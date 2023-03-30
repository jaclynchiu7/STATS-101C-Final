### LOADING LIBRARIES
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(stringr)
library(tidyr)
library(leaps)
library(randomForest)
library(gbm)
library(glmnet)
library(pls)

### LOADING DATA 
train <- read.csv("data/training.csv")
test <- read.csv("data/test.csv")

### BREAKING DOWN PUBLISHED DATE INTO ADDITIONAL VARIABLES
# For Train Data
train$PublishedDate <- paste0("0", train$PublishedDate)
time1 <- str_split_fixed(train$PublishedDate, " ", 2)
train$date <- time1[,1]
train$time <- time1[,2]
time2 <- str_split_fixed(train$date, "/", 3)
time3 <- str_split_fixed(train$time, ":", 2)
train$month <- time2[,1]
train$day <- time2[,2]
train$year <- time2[,3]
train$hour <- time3[,1]
train$minutes <- time3[,2]

n <- nrow(train)
c <- which(colnames(train) == "day")

for (i in 1:n){
  if (nchar(train[i, c]) == 1){
    train[i, c] <- paste0("0", train[i, c])
  }
  else{
    train[i,c] <- train[i,c]
  }
}

train$date <- paste(train$year, train$month, train$day, sep = "-")
train$week <- as.numeric(wday(train$date))
train$minutes <- as.numeric(train$minutes)
train$hour <- as.numeric(train$hour)
train$month <- as.numeric(train$month)
train$day <- as.numeric(train$day)
train$year <- as.numeric(train$year)
delete_char <- c("PublishedDate", "date", "time")

delete_cols <- match(delete_char, names(train))
train <- train[, -delete_cols]

# For Test Data
test$PublishedDate <- paste0("0", test$PublishedDate)
time1 <- str_split_fixed(test$PublishedDate, " ", 2)
test$date <- time1[,1]
test$time <- time1[,2]
time2 <- str_split_fixed(test$date, "/", 3)
time3 <- str_split_fixed(test$time, ":", 2)
test$month <- time2[,1]
test$day <- time2[,2]
test$year <- time2[,3]
test$hour <- time3[,1]
test$minutes <- time3[,2]

c <- which(colnames(test) == "day")
n <- nrow(test)

for (i in 1:n){
  if (nchar(test[i, c]) == 1){
    test[i, c] <- paste0("0", test[i, c])
  }
  else{
    test[i,c] <- test[i,c]
  }
}

test$date <- paste(test$year, test$month, test$day, sep = "-")
test$week <- as.numeric(wday(test$date))
test$minutes <- as.numeric(test$minutes)
test$hour <- as.numeric(test$hour)
test$month <- as.numeric(test$month)
test$day <- as.numeric(test$day)
test$year <- as.numeric(test$year)
delete_cols <- match(delete_char, names(test))
test <- test[, -delete_cols]

### BINARY FACTOR CONSOLIDATION
# Combine factors for Num_Subscribers_Base
Num_Subscribers_Base <- character()
low <- which(train$Num_Subscribers_Base_low == 1)
low_mid <- which(train$Num_Subscribers_Base_low_mid == 1)
mid_high <- which(train$Num_Subscribers_Base_mid_high == 1)
high <- which(train$Num_Subscribers_Base_low == 0 & train$Num_Subscribers_Base_low_mid == 0 &
                train$Num_Subscribers_Base_mid_high == 0)

Num_Subscribers_Base[low] <- "low"
Num_Subscribers_Base[low_mid] <- "low mid"
Num_Subscribers_Base[mid_high] <- "mid high"
Num_Subscribers_Base[high] <- "high"
train$Num_Subscribers_Base <- as.factor(Num_Subscribers_Base)

# Combine factors for Num_Views_Base
Num_Views_Base <- character()
low <- which(train$Num_Views_Base_low == 1)
low_mid <- which(train$Num_Views_Base_low_mid == 1)
mid_high <- which(train$Num_Views_Base_mid_high == 1)
high <- which(train$Num_Views_Base_low == 0 & train$Num_Views_Base_low_mid == 0 &
                train$Num_Views_Base_mid_high == 0)

Num_Views_Base[low] <- "low"
Num_Views_Base[low_mid] <- "low mid"
Num_Views_Base[mid_high] <- "mid high"
Num_Views_Base[high] <- "high"
train$Num_Views_Base <- as.factor(Num_Views_Base)

# Combine factors for Avg_Growth
avg_growth <- character()
low <- which(train$avg_growth_low == 1)
low_mid <- which(train$avg_growth_low_mid == 1)
mid_high <- which(train$avg_growth_mid_high == 1)
high <- which(train$avg_growth_low == 0 & train$avg_growth_low_mid == 0 & train$avg_growth_mid_high ==
                0)
avg_growth[low] <- "low"
avg_growth[low_mid] <- "low mid"
avg_growth[mid_high] <- "mid high"
avg_growth[high] <- "high"
train$avg_growth <- as.factor(avg_growth)

# Combine factors for count_vids
count_vids <- character()
low <- which(train$count_vids_low == 1)
low_mid <- which(train$count_vids_low_mid == 1)
mid_high <- which(train$count_vids_mid_high == 1)
high <- which(train$count_vids_low == 0 & train$count_vids_low_mid == 0 & train$count_vids_mid_high == 0)
count_vids[low] <- "low"
count_vids[low_mid] <- "low mid"
count_vids[mid_high] <- "mid high"
count_vids[high] <- "high"
train$count_vids <- as.factor(count_vids)
delete_factors <- c("Num_Subscribers_Base_low", "Num_Subscribers_Base_low_mid", "Num_Subscribers_Base_mid_high", "Num_Views_Base_low", "Num_Views_Base_low_mid", "Num_Views_Base_mid_high", "avg_growth_low",
                    "avg_growth_low_mid", "avg_growth_mid_high", "count_vids_low", "count_vids_low_mid",
                    "count_vids_mid_high")
delete_cols<- match(delete_factors, names(train))
train <- train[, -delete_cols]

# Repeat above but for the testing data
Num_Subscribers_Base <- character()
low <- which(test$Num_Subscribers_Base_low == 1)
low_mid <- which(test$Num_Subscribers_Base_low_mid == 1)
mid_high <- which(test$Num_Subscribers_Base_mid_high == 1)
high <- which(test$Num_Subscribers_Base_low == 0 & test$Num_Subscribers_Base_low_mid == 0 &
                test$Num_Subscribers_Base_mid_high == 0)
Num_Subscribers_Base[low] <- "low"
Num_Subscribers_Base[low_mid] <- "low mid"
Num_Subscribers_Base[mid_high] <- "mid high"
Num_Subscribers_Base[high] <- "high"

test$Num_Subscribers_Base <- as.factor(Num_Subscribers_Base)
Num_Views_Base <- character()
low <- which(test$Num_Views_Base_low == 1)
low_mid <- which(test$Num_Views_Base_low_mid == 1)
mid_high <- which(test$Num_Views_Base_mid_high == 1)
high <- which(test$Num_Views_Base_low == 0 & test$Num_Views_Base_low_mid == 0 &
                test$Num_Views_Base_mid_high == 0)
Num_Views_Base[low] <- "low"
Num_Views_Base[low_mid] <- "low mid"
Num_Views_Base[mid_high] <- "mid high"
Num_Views_Base[high] <- "high"
test$Num_Views_Base <- as.factor(Num_Views_Base)
avg_growth <- character()
low <- which(test$avg_growth_low == 1)
low_mid <- which(test$avg_growth_low_mid == 1)
mid_high <- which(test$avg_growth_mid_high == 1)
high <- which(test$avg_growth_low == 0 & test$avg_growth_low_mid == 0 & test$avg_growth_mid_high == 0)
avg_growth[low] <- "low"
avg_growth[low_mid] <- "low mid"
avg_growth[mid_high] <- "mid high"
avg_growth[high] <- "high"
test$avg_growth <- as.factor(avg_growth)
count_vids <- character()
low <- which(test$count_vids_low == 1)
low_mid <- which(test$count_vids_low_mid == 1)
mid_high <- which(test$count_vids_mid_high == 1)
high <- which(test$count_vids_low == 0 & test$count_vids_low_mid == 0 & test$count_vids_mid_high == 0)
count_vids[low] <- "low"
count_vids[low_mid] <- "low mid"
count_vids[mid_high] <- "mid high"
count_vids[high] <- "high"
test$count_vids <- as.factor(count_vids)
delete_factors <- c("Num_Subscribers_Base_low", "Num_Subscribers_Base_low_mid", "Num_Subscribers_Base_mid_high", "Num_Views_Base_low", "Num_Views_Base_low_mid", "Num_Views_Base_mid_high", "avg_growth_low",
                    "avg_growth_low_mid", "avg_growth_mid_high", "count_vids_low", "count_vids_low_mid",
                    "count_vids_mid_high")
delete_cols<- match(delete_factors, names(test))
test <- test[, -delete_cols]

# Save preprocessed csv 
write.csv(train, "data/preprocessed_train.csv", row.names=FALSE)
write.csv(test, "data/preprocessed_test.csv", row.names=FALSE)
