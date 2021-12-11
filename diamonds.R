# Clear All Variables & Clear Screen
rm(list=ls())
cat("\014")

# Install packages 
library(caTools)
library(rpart)
library(rpart.plot)
library(tictoc)
library(randomForest)
library(ranger)
library(caret)
library(e1071)


# Setting the same random seed
set.seed(1234)

# Load the dataset 
Gem = read.csv("diamonds.csv")
str(Gem)

# Splitting the dataset into training and testing
split <- sample.split(Gem$price, SplitRatio = 0.8)
head (split)
Train  = subset(Gem, split==TRUE)
Test = subset(Gem, split==FALSE)

# Build a classification tree CART model on the training data
tic()
Gem.CT = rpart(price ~ carat + cut + color + clarity + depth + table, data = Train, method = "class", minbucket = )
Gem.CT = rpart()
toc()

# Plot the tree
prp(Gem.CT)

# Make prediction on the test set
Gem.CT.predTest = predict(Gem.CT, newdata = Test, type = "class")

# Create the confusion matrix
CT.predTable <- table(Test$price, Gem.CT.predTest)
CT.predTable

sum(diag(CT.predTable))/sum(CT.predTable)

# Random Forest
tr_control<- trainControl(method='cv', number=2, search = 'random')

# Training a Random Forest
set.seed(1234)
tic()
rf_train <- train(price~carat+cut+color+clarity+depth+table,
      data=Train, 
      metric = ("Rsquared", "RMSE"),
      method = "rf",
      trControl = tr_control)
toc()
print(rf_train)
## Train RMSE is 677.94 and R Squared is .973
GemForest = ranger(price ~ carat + cut + color + clarity + depth + table, data = Train,
                   num.trees = 250,
                   mtry=sqrt(ncol(Train)),
                   min.node.size = 5,
                   respect.unordered.factors = 'order')
GemForest = ranger(price ~ carat + cut + color + clarity + depth + table, data = Train, num.trees = 200, min.node.size=15)
summary(GemForest)
GemPredictForest = predict(GemForest, data=Test) # automatically assumes threshold = 0.5 and directly predicts 0 or 1

residual<-GemPredictForest$predictions - Test$price
rmse<-sqrt(mean(residual^2))
print(rmse)

SSE <- sum((Test$price -GemPredictForest$predictions)^2)
SST<- sum((Test$price - mean(Train$price))^2)
GemForestr2<- 1- SSE/SST
print(GemForestr2)
# Out of Sample RMSE is 505.16 and R Squared is 0.981

