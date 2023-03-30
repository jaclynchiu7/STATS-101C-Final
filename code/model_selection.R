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

### LOAD PREPROCESSED DATA
train <- read.csv("data/preprocessed_train.csv")
test <- read.csv("data/preprocessed_test.csv")


### STANDARDIZING PREDICTORS
train <- na.omit(train)
nonnumeric <- c(which(unlist(lapply(train, class)) == "character"), which(unlist(lapply(train, class))
                                                                          == "factor"))
nonnumeric_col <- match(c("id", "Num_Subscribers_Base", "Num_Views_Base", "avg_growth", "count_vids", "growth_2_6"), names(train))
nonnumeric_col
mu <- apply(train[, -nonnumeric_col], 2, mean)
dev <- apply(train[, -nonnumeric_col], 2, sd)

# standardize train data
train2 <- sweep(train[ ,-nonnumeric_col], 2, mu, '-')
train2 <- sweep(train2, 2, dev, '/')
train2 <- data.frame("id" = train$id, train2, "Num_Subscribers_Base" = train$Num_Subscribers_Base,
                     "Num_Views_Base" = train$Num_Views_Base, "avg_growth" = train$avg_growth, "count_vids" = train$count_vids, "growth_2_6" = train$growth_2_6)

# standardize test data
nonnumeric <- c(which(unlist(lapply(test, class)) == "character"), which(unlist(lapply(test, class)) ==
                                                                           "factor"))
nonnumeric_col <- match(c("id", "Num_Subscribers_Base", "Num_Views_Base", "avg_growth", "count_vids"),
                        names(test))
nonnumeric_col
test2 <- sweep(test[ ,-nonnumeric_col], 2, mu, '-')
test2 <- sweep(test2, 2, dev, '/')
test2 <- data.frame("id" = test$id, test2, "Num_Subscribers_Base" = test$Num_Subscribers_Base,
                    "Num_Views_Base" = test$Num_Views_Base, "avg_growth" = test$avg_growth, "count_vids"= test$count_vids)

### DETERMINING BEST STATISTICAL MODEL
# Create RMSE function
get_rmse <- function(pred, true){
  return(sqrt(mean((pred - true)^2)))
}

# Linear Regression
set.seed(123)
i <- 1:nrow(train2)
i_train <- sample(i, 0.7*nrow(train2), replace = F)
training <- train2[i_train, ] # Approximately 70% of data is training set
testing <- train2[-i_train, ] # Approximately 30% of data is test set
dim(training)
dim(testing)
lm <- lm(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 + cnn_86 + cnn_88 +
           cnn_89 +
           punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
           Num_Subscribers_Base +
           Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
           Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training)
lm_pred <- predict(lm, testing)
lm_rmse <- get_rmse(lm_pred, testing$growth_2_6)

# Ridge Regression
train_x <- model.matrix(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 +
                          cnn_86 + cnn_88 + cnn_89 +
                          punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month +
                          hour + Num_Subscribers_Base +
                          Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base
                        +
                          Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, training)[,-1]
test_x <- model.matrix(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 +
                         cnn_86 + cnn_88 + cnn_89 +
                         punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour
                       + Num_Subscribers_Base +
                         Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
                         Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, testing)[,-1]
y <- training$growth_2_6
i.exp <- seq(10, -2, length = 100)
grid <- 10^i.exp
ridge.mod <- glmnet(train_x, y, family = "gaussian", alpha = 0,
                    lambda = grid, standardize = TRUE)
cv.output <- cv.glmnet(train_x, y, family = "gaussian", alpha = 0,
                       lambda = grid, standardize = TRUE,
                       nfolds = 10)
best.lambda.cv <- cv.output$lambda.min
rr.pred <- predict(ridge.mod, s = best.lambda.cv, newx = test_x, type = "response")
ridge_rmse <- get_rmse(rr.pred, testing$growth_2_6)

# LASSO
lasso.mod <- glmnet(train_x, y, family = "gaussian", alpha = 1,
                    lambda = grid, standardize = TRUE)
lasso.cv.output <- cv.glmnet(train_x, y, family = "gaussian", alpha = 1,
                             lambda = grid, standardize = TRUE,
                             nfolds = 10)
lasso.best.lambda.cv <- lasso.cv.output$lambda.min
lasso.pred <- predict(lasso.mod, s = lasso.best.lambda.cv, newx = test_x, type = "response")
lasso_rmse <- get_rmse(lasso.pred, testing$growth_2_6)

# Elastic Net
enet.mod <- glmnet(train_x, y, family = "gaussian", alpha = 0.5,
                   lambda = grid, standardize = TRUE)
enet.cv.output <- cv.glmnet(train_x, y, family = "gaussian", alpha = 0.5,
                            lambda = grid, standardize = TRUE,
                            nfolds = 10)
enet.best.lambda.cv <- enet.cv.output$lambda.min
enet.pred <- predict(enet.mod, s = enet.best.lambda.cv, newx = test_x, type = "response")
elastic_rmse <- get_rmse(enet.pred, testing$growth_2_6)

# PCA
pcr_mod <- pcr(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 + cnn_86 +
                 cnn_88 + cnn_89 +
                 punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
                 Num_Subscribers_Base +
                 Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
                 Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training, scale =
                 FALSE, validation = "CV")
summary(pcr_mod)
validationplot(pcr_mod, val.type = "MSEP")
pcr <- pcr(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 + cnn_86 + cnn_88
           + cnn_89 +
             punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
             Num_Subscribers_Base +
             Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
             Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids,
           data = training, scale = FALSE, ncomp = 31)
pcr_pred <- predict(pcr, testing)
pcr_rmse <- get_rmse(pcr_pred, testing$growth_2_6)

# PLS
pls_model = plsr(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 + cnn_86 +
                   cnn_88 + cnn_89 +
                   punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
                   Num_Subscribers_Base +
                   Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
                   Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training, scale =
                   FALSE, validation = "CV")
summary(pls_model)
validationplot(pls_model, val.type = "MSEP")
pls_mod <- plsr(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 + cnn_86 +
                  cnn_88 + cnn_89 +
                  punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
                  Num_Subscribers_Base +
                  Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
                  Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training, scale =
                  FALSE, ncomp = 18)
summary(pls_mod)
pls_pred <- predict(pls_mod, testing)
pls_rmse <- get_rmse(pls_pred, testing$growth_2_6)

# Bagging
bag_mod <- randomForest(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 +
                          cnn_86 + cnn_88 + cnn_89 +
                          punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month +
                          hour + Num_Subscribers_Base +
                          Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base
                        +
                          Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training,
                        mtry = 19, importance = TRUE)
sqrt(bag_mod$mse[500]) # out of bag rmse
bag_pred <- predict(bag_mod, testing)
bag_rmse <- get_rmse(bag_pred, testing$growth_2_6)

# Random Forest
rf_mod <- randomForest(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 +
                         cnn_86 + cnn_88 + cnn_89 +
                         punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour
                       + Num_Subscribers_Base +
                         Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
                         Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training,
                       mtry = 6, importance = TRUE)
sqrt(rf$mse[500]) # out of bag rmse
rf_pred <- predict(rf_mod, testing)
rf_rmse <- get_rmse(rf_pred, testing$growth_2_6)

# Boosting
ntrees <- c(500, 750, 1000)
i.exp <- seq(-10, 0, length = 100)
grid <- 10^i.exp
results = matrix(rep(0, length(ntrees)*length(grid)),length(ntrees),length(grid))
rownames(results) = apply(as.matrix(ntrees), 2, function(t){return(paste("n =",t))})
colnames(results) = apply(as.matrix(grid), 2, function(t){return(paste("lambda =",t))})
for (i1 in 1:length(ntrees))
{
  n = ntrees[i1]
  for (i2 in 1:length(grid))
  {
    shrink = grid[i2]
    boost_mod <- gbm(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 +
                       cnn_86 + cnn_88 + cnn_89 +
                       punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
                       Num_Subscribers_Base +
                       Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
                       Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training,
                     distribution = "gaussian",
                     n.trees = n, shrinkage = shrink)
    boost_preds = predict(boost_mod, testing)
    results[i1,i2] <- get_rmse(boost_preds, testing$growth_2_6)
  }
}

# Based on cross-validation to tune parameters: a lambda of 0.0774263682681128 and 1000 trees gave the lowest rmse
boost_mod <- gbm(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 + cnn_86 +
                   cnn_88 + cnn_89 +
                   punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
                   Num_Subscribers_Base +
                   Num_Views_Base + avg_growth + count_vids + Num_Subscribers_Base:Num_Views_Base +
                   Num_Subscribers_Base:count_vids + Num_Views_Base:count_vids, data = training,
                 distribution = "gaussian",
                 n.trees = 1000, shrinkage = 0.0774263682681128)
boost_pred <- predict(boost_mod, testing)
boost_rmse <- get_rmse(boost_pred, testing$growth_2_6)

### FINAL MODELING
# Tune 19 predictor model
mtry_vals = 1:19
nodesize_vals = c(2,5,10,20)
results = matrix(rep(0, length(mtry_vals)*length(nodesize_vals)),length(mtry_vals),length(nodesize_vals
))
rownames(results) = apply(as.matrix(mtry_vals), 2, function(t){return(paste("m =",t))})
colnames(results) = apply(as.matrix(nodesize_vals), 2, function(t){return(paste("n =",t))})
for (i1 in 1:length(mtry_vals))
{
  m = mtry_vals[i1]
  for (i2 in 1:length(nodesize_vals))
  {
    n = nodesize_vals[i2]
    rf_model = randomForest(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25
                            + cnn_86 + cnn_88 + cnn_89 +
                              punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month +
                              hour + Num_Subscribers_Base +
                              Num_Views_Base + avg_growth + count_vids, data = training, ntree = 1000,
                            mtry = mtry_vals, nodesize = nodesize_vals, importance = TRUE)
    rf_preds = predict(rf_model, testing)
    results[i1,i2] <- get_rmse(rf_preds, testing$growth_2_6)
  }
}
# Result that m = 7, nodesize = 10 yields lowest RMSE
# Tune 16 predictor model
mtry_vals = 1:16
nodesize_vals = c(2,5,10,20)
results = matrix(rep(0, length(mtry_vals)*length(nodesize_vals)),length(mtry_vals),length(nodesize_vals
))
rownames(results) = apply(as.matrix(mtry_vals), 2, function(t){return(paste("m =",t))})
colnames(results) = apply(as.matrix(nodesize_vals), 2, function(t){return(paste("n =",t))})
for (i1 in 1:length(mtry_vals))
{
  m = mtry_vals[i1]
  for (i2 in 1:length(nodesize_vals))
  {
    n = nodesize_vals[i2]
    rf_model = randomForest(growth_2_6 ~ Duration + views_2_hours + cnn_10 + cnn_12 + cnn_25 + cnn_86 +
                              cnn_88 + cnn_89 +
                              punc_num_..28 + num_uppercase_chars + month + hour + Num_Subscribers_Base +
                              Num_Views_Base + avg_growth + count_vids, data = training, ntree = 1000,
                            mtry = mtry_vals, nodesize = nodesize_vals, importance = TRUE)
    rf_preds = predict(rf_model, testing)
    results[i1,i2] <- get_rmse(rf_preds, testing$growth_2_6)
  }
}
# Result suggests m = 14, n = 5 gives lowest RMSE, but we chose m = 6 and n = 10 instead to take advantage of the benefits of Random Forest
# Cross Validation with 19 Predictors
set.seed(1)
train2<-train2[sample(nrow(train)),]
get_rmse <- function(pred, true){
  return(sqrt(mean((pred - true)^2)))
}
#Create n equally size folds
folds <- cut(seq(1,nrow(train2)),breaks=5,labels=FALSE)
part_rmse <- c()
for(i in 1:5){
  #Segment your data by fold using the which() function
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- train2[testIndexes, ]
  trainData <- train2[-testIndexes, ]
  #Use the test and train data partitions however you desire...
  rf <- randomForest(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 +
                       cnn_86 + cnn_88 + cnn_89 +
                       punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
                       Num_Subscribers_Base +
                       Num_Views_Base + avg_growth + count_vids, data = trainData, ntree = 1000, mtry =
                       7, nodesize = 10, importance = TRUE)
  preds <- predict(rf, testData)
  part_rmse[i] <- get_rmse(preds, testData$growth_2_6)
}
rmse <- mean(part_rmse)
print(part_rmse)
print(rmse)

#Out of Bag Error 19 Predictors
rf <- randomForest(growth_2_6 ~ Duration + views_2_hours + hog_454 + cnn_10 + cnn_12 + cnn_25 + cnn_86
                   + cnn_88 + cnn_89 +
                     punc_num_..28 + num_words + num_uppercase_chars + num_digit_chars + month + hour +
                     Num_Subscribers_Base +
                     Num_Views_Base + avg_growth + count_vids, data = train2, ntree = 1000, mtry = 7,
                   nodesize = 10, importance = TRUE)
sqrt(rf$mse[1000])
plot(rf, main = "Random Forest Model")
varImpPlot(rf, scale = TRUE, main = "Random Forest Model")

# Cross Validation with 16 Predictors
set.seed(1)
train2<-train2[sample(nrow(train)),]
get_rmse <- function(pred, true){
  return(sqrt(mean((pred - true)^2)))
}

#Create n equally size folds
folds <- cut(seq(1,nrow(train2)),breaks=5,labels=FALSE)
part_rmse <- c()
for(i in 1:5){
  #Segment your data by fold using the which() function
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- train2[testIndexes, ]
  trainData <- train2[-testIndexes, ]
  #Use the test and train data partitions however you desire...
  rf <- randomForest(growth_2_6 ~ Duration + views_2_hours + cnn_10 + cnn_12 + cnn_25 + cnn_86 + cnn_88
                     + cnn_89 +
                       punc_num_..28 + num_uppercase_chars + month + hour + Num_Subscribers_Base +
                       Num_Views_Base + avg_growth + count_vids, data = trainData, ntree = 1000, mtry =
                       6, nodesize = 10, importance = TRUE)
  preds <- predict(rf, testData)
  part_rmse[i] <- get_rmse(preds, testData$growth_2_6)
}
rmse <- mean(part_rmse)
print(part_rmse)
print(rmse)

#Out of Bag Error 16 Predictors
rf <- randomForest(growth_2_6 ~ Duration + views_2_hours + cnn_10 + cnn_12 + cnn_25 + cnn_86 + cnn_88 +
                     cnn_89 +
                     punc_num_..28 + num_uppercase_chars + month + hour + Num_Subscribers_Base +
                     Num_Views_Base + avg_growth + count_vids, data = trainData, ntree = 1000, mtry = 6,
                   nodesize = 10, importance = TRUE)
sqrt(rf$mse[1000])
### LOOKING AT MOST SIGNIFICANT PREDICTORS IN THE MODEL
importance(rf)


### PLOTS FOR REPORT
# Table of all RMSE values across different models
models <- c("Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net",
            "Principal Component Analysis", "Partial Least Squares", "Bagging", "Random Forest", "Boosting")
table <- data.frame(lm_rmse, ridge_rmse, lasso_rmse, elastic_rmse, pcr_rmse, pls_rmse, bag_rmse,
                    rf_rmse, boost_rmse)
names(table) <- models
knitr::kable(table, caption = "RMSE for Different Models")
# Variable Importance Plot
varImpPlot(rf, scale = TRUE)