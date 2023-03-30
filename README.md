# How to Grow Your YouTube Channel 
Youtube is a large scale video-sharing platform that can a source of money for people aspiring to be content creators. The amount of views a video receives is important in determining how much revnue a content creator will earn and the overall health of a channel. We evaulate the ability to predict the percent change in views on a video between the second and sixth hour since publishing. Our methodology includes preprocessing and transforming predictor variables, finding significant predictors, and applying a variety of regression models. 


# Model Accuracies

Model | RMSE
--- | ---
Linear Regression | 1.6397
Ridge Regression | 1.6404
Lasso Regression | 1.6479
Elastic Net | 1.6443
Principal Component Analysis | 1.8554
Partial Least Squares | 1.7049 
Bagging | 1.4459
**Random Forest** | **1.4457**
Boosting | 1.6127

# Plot of important variables from Random Forest Model 

<img width = "900" height = "350" src = "https://user-images.githubusercontent.com/72675054/228721995-e74223f7-a755-432d-b6a8-b4b640a4147c.png">

# Takeaways
* Random Forest model performed the best with 19 predictors 
* Most important predictors included average growth of video, high-level imag efeatures from thumbnail image of the vdeo, and number of subscribers and view to a video's channel 
