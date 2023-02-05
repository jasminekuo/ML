# Machine Learning Final Project

# load required packages
library(tidyverse)
library(ggplot2)
library(ggrepel)
library(gridExtra)
library(factoextra)
library(GGally)
library(MASS)
library(fpc)
library(caret)
library(palmerpenguins)
library(foreign)
library(e1071)
library(rpart)
library(rpart.plot)
library(NeuralNetTools)
library(randomForest)
library(readstata13)
if(!require(parttree)) remotes::install_github("grantmcdermott/parttree")
library(parttree)

# load data
load("Project2.Rsav")

# SVM

# Problem: Linear Kernel

# set seed 
set.seed(2011001)

# train svm: 10-fold cross-validation
svm.linear.tc <- trainControl(method="cv",
                              number=10)

svm.linear <- train(region ~., 
                    data = training, 
                    method = "svmLinear2", 
                    trControl = svm.linear.tc
)

# parameters of best model: 0.5
svm.linear$bestTune


# refit the best model with svm
fit.lin <-svm(region~.,
              data=training,
              kernel="linear", 
              cost = 0.50)


# visualize plot on final model 
plot(fit.lin,
     training,
     arachidi~palmitic,
     slice=list(stearic=mean(training$stearic),
                linoleni=mean(training$linoleni),
                palmitol=mean(training$palmitol)
     ))


# predict on test data 
svm.linear.pred <- predict(fit.lin, newdata = testing)

# confusion matrix 
confusionMatrix(table(svm.linear.pred, testing$region))$table

# accuracy 
confusionMatrix(table(svm.linear.pred, testing$region))$overall['Accuracy']

# Problem: Radial Kernel

# set seed 
set.seed(2011001)

# train svm: 10-fold cross-validation
svm.radial.tc <- trainControl(method="cv",
                              number=10)

svm.radial <- train(region ~., 
                    data = training, 
                    method = "svmRadial", 
                    trControl = svm.radial.tc
)

svm.radial$results

# refit the best model 
fit.radial <-svm(region~.,
                 data=training,
                 kernel="radial",
                 cost=1)

# visualize plot on final model 
plot(fit.radial,
     training,
     arachidi~palmitic,
     slice=list(stearic=mean(training$stearic),
                linoleni=mean(training$linoleni),
                palmitol=mean(training$palmitol)
     ))


# predict on test data 
svm.radial.pred <- predict(fit.radial, newdata = testing)

# confusion matrix 
confusionMatrix(table(svm.radial.pred, testing$region))$table

# accuracy 
confusionMatrix(table(svm.radial.pred, testing$region))$overall['Accuracy']

# Regression Trees

# Problem: rpart

# step 0: set seed
set.seed(2011001)

# step 1: train regression tree model using caret
fit3 <- train(region ~ ., data = training, method = "rpart2", 
              trControl = trainControl(method = "cv", number = 10))
fit3

# step 2: extract best model
best_model3 <- fit3$finalModel

# step 3: visualize best model
rpart.plot(best_model3)

# step 4: predict on TEST data
predict3 <- predict(best_model3, newdata = testing, type = "class")

confusionMatrix(predict3, testing$region)

# Problem: randomForest

# step 0: set seed
set.seed(2011001)

# step 1: train random forest model using caret
fit4 <- train(region ~ ., data = training, method = "rf", 
              trControl = trainControl(method = "cv", number = 10), importance = TRUE)
fit4

# step 2: extract best model
best_model4 <- fit4$finalModel

# step 3: compute average tree size
treesize(best_model4) %>%
        mean()

# step 4: determine most predictive variables
par(mfrow=c(1,3))
for (i in 1:3) {
        varImpPlot(best_model4, type = 1, class = best_model4$classes[i], 
                   main = "Olive Oil RF", sub = "Mean Decrease Accuracy (%)", 
                   cex = .5)
}

# step 5: predict on TEST data
predict4 <- predict(best_model4, newdata = testing)
confusionMatrix(predict4, testing$region)