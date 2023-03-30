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

### ELIMINATE HIGHLY CORRELATED PREDICTORS
nonnumeric <- c(which(unlist(lapply(train, class)) == "character"), which(unlist(lapply(train, class))
                                                                          == "factor"))
training <- train[, -nonnumeric]
training <- training[, -which(apply(training, 2, sd) == 0)]
var_names <- names(training)
num_vars = length(names(training))
for (i1 in 1:(num_vars-1)){
  v1 = var_names[i1]
  for (i2 in (i1+1):num_vars)
  {
    v2 = var_names[i2]
    c = cor(training[,v1], training[,v2])
    if (abs(c) > 0.8)
    {
      print(paste(v1, v2, sep=' - '))
      print(c)
    }
  }
}

delete_cor <- c("hog_13", "hog_40", "hog_132", "hog_330", "hog_133", "hog_304", "hog_303", "hog_165", "
hog_336", "hog_363", "hog_166", "hog_337", "hog_364", "hog_204", "hog_375", "hog_402", "hog_205", "
hog_376", "hog_403", "hog_442", "hog_469", "hog_306", "hog_477",
                "hog_314", "hog_485", "hog_512", "hog_378", "hog_549", "hog_576", "hog_386", "hog_557",
                "hog_584", "hog_526", "hog_697", "hog_724", "cnn_17", "cnn_25", "sd_pixel_val", "mean_red", "mean_green", "sd_green", "mean_blue", "sd_red", "sd_blue",
                "punc_num_..9", "punc_num_..24", "hog_774", "hog_782", "hog_108", "hog_116", "hog_152",
                "hog_182", "hog_183", "hog_686", "hog_702", "hog_703", "hog_704", "hog_738", "hog_746", "hog_810", "hog_818", "hog_852",
                "hog_815", "hog_144", "hog_514", "hog_259", "hog_705", "hog_725", "hog_743")

delete_cols<- match(delete_cor, names(train))
train <- train[, -delete_cols]
delete_cols <- match(delete_cor, names(test))
test <- test[, -delete_cols]

### RUN LINEAR REGRESSION MODEL TO DETERMINE SIGNIFICANT PREDICTORS
sig_mod <- lm(growth_2_6~., data = train)
summary(sig_mod)
var_names <- c("Duration", "hog_11", "hog_454", "hog_492", "hog_495", "hog_522", "hog_641", "hog_665",
               "hog_675", "hog_677", "hog_716", "hog_855", "cnn_9",
               "cnn_10", "cnn_12", "cnn_86", "cnn_88", "cnn_89", "doc2vec_2", "doc2vec_7", "doc2vec_11",
               "doc2vec_12", "doc2vec_13", "punc_num_..1",
               "punc_num_..3", "punc_num_..8", "punc_num_..11", "punc_num_..14", "punc_num_..15", "punc_num_..20", "punc_num_..28", "num_words",
               "num_stopwords", "num_uppercase_words", "month", "Num_Subscribers_Base", "Num_Views_Base", "avg_growth", "count_vids", "growth_2_6")
predictor_cols<- match(var_names, names(train))
training2 <- train[, predictor_cols]

### ELIMINATE HIGHLY CORRELATED PREDICTORS AGAIN
nonnumeric <- c(which(unlist(lapply(training2, class)) == "character"), which(unlist(lapply(training2,
                                                                                            class)) == "factor"))
training2 <- training2[, -nonnumeric]
var_names <- names(training2)
num_vars = length(names(training2))
for (i1 in 1:(num_vars-1)){
  v1 = var_names[i1]
  for (i2 in (i1+1):num_vars)
  {
    v2 = var_names[i2]
    c = cor(training2[,v1], training2[,v2])
    if (abs(c) > 0.8)
    {
      print(paste(v1, v2, sep=' - '))
      print(c)
    }
  }
}

# Remove hog_522:
var_names <- c("Duration", "hog_11", "hog_454", "hog_492", "hog_495", "hog_641", "hog_665", "hog_675",
               "hog_677", "hog_716", "hog_855", "cnn_9",
               "cnn_10", "cnn_12", "cnn_86", "cnn_88", "cnn_89", "doc2vec_2", "doc2vec_7", "doc2vec_11",
               "doc2vec_12", "doc2vec_13", "punc_num_..1",
               "punc_num_..3", "punc_num_..8", "punc_num_..11", "punc_num_..14", "punc_num_..15", "punc_num_..20", "punc_num_..28", "num_words",
               "num_stopwords", "num_uppercase_words", "month", "Num_Subscribers_Base", "Num_Views_Base", "avg_growth", "count_vids", "growth_2_6")
predictor_cols<- match(var_names, names(train))
training3 <- train[, predictor_cols]

### USE LASSO TO FIND PREDICTORS
set.seed(123)
i <- 1:nrow(training3)
i_train <- sample(i, 0.7*nrow(training3), replace = F)
training <- training3[i_train, ] # Approximately 70% of data is training set
testing <- training3[-i_train, ] # Approximately 30% of data is test set
train_x <- model.matrix(growth_2_6 ~ ., training)[,-1]
test_x <- model.matrix(growth_2_6 ~ ., testing)[,-1]
y <- training$growth_2_6
i.exp <- seq(10, -2, length = 100)
grid <- 10^i.exp
lasso.mod <- glmnet(train_x, y, family = "gaussian", alpha = 1,
                    lambda = grid, standardize = TRUE)
lasso.cv.output <- cv.glmnet(train_x, y, family = "gaussian", alpha = 1,
                             lambda = grid, standardize = TRUE,
                             nfolds = 10)
lasso.best.lambda.cv <- lasso.cv.output$lambda.min
predict(lasso.mod, s = lasso.best.lambda.cv, newx = test_x, type = "coefficients")

### USE BEST SUBSET SELECTION TO FIND PREDICTORS
best_subset <- regsubsets(growth_2_6~., data = training3, nbest = 1, nvmax = 25,
                          intercept = TRUE, method = "exhaustive",
                          really.big = TRUE)
sumBS <- summary(best_subset)


### INVESTIGATE IF THERE ARE INTERACTIONS BETWEEN CATEGORICAL VARIABLES
p1 <- ggplot() +
  aes(x = train$Num_Subscribers_Base, color = train$Num_Views_Base, group = train$Num_Views_Base, y =
        train$growth_2_6) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line")
p2 <- ggplot() +
  aes(x = train$Num_Subscribers_Base, color = train$avg_growth, group = train$avg_growth, y =
        train$growth_2_6) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line")
p3 <- ggplot() +
  aes(x = train$Num_Subscribers_Base, color = train$count_vids, group = train$count_vids, y =
        train$growth_2_6) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line")
p4 <- ggplot() +
  aes(x = train$Num_Views_Base, color = train$avg_growth, group = train$avg_growth, y =
        train$growth_2_6) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line")
p5 <- ggplot() +
  aes(x = train$Num_Views_Base, color = train$count_vids, group = train$count_vids, y =
        train$growth_2_6) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line")
p6 <- ggplot() +
  aes(x = train$avg_growth, color = train$count_vids, group = train$count_vids, y = train$growth_2_6) +
  stat_summary(fun.y = mean, geom = "point") +
  stat_summary(fun.y = mean, geom = "line")
# There are interactions between num_subscribers_base and num_views_base, num_subscribers base and count_vids, num_views base and count_vids

