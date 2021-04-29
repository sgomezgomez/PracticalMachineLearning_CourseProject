## Loading packages and relevant dependencies
## This code chunk will not be displayed on the R Markdown document
library(gbm)
library(caret)
library(rattle)
library(e1071)
library(kernlab)
library(olsrr)
library(rpart)
library(rpart.plot)
library(nlme)
library(mgcv)

##download.file(url = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', 'pml-training.csv', method = 'curl')
##download.file(url = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', 'pml-testing.csv', method = 'curl')
pmltraining = read.table('pml-training.csv', sep = ',', header = TRUE, na.strings = c('NA', '#DIV/0!', ''))
pmltesting = read.table('pml-testing.csv', sep = ',', header = TRUE, na.strings = c('NA', '#DIV/0!', ''))
pmltraining$classe = as.factor(pmltraining$classe)
pmltraining$cvtd_timestamp = as.Date(pmltraining$cvtd_timestamp, "%d/%m/%Y %H:%M")

## Setting seed for reproducibility purposes
set.seed(100)
## Creating model training partition
modTrainingPartition = createDataPartition(y = pmltraining$classe, p = 0.75, list = FALSE)
pmlmodeltraining = pmltraining[modTrainingPartition, ]
pmlmodeltesting = pmltraining[-modTrainingPartition, ]
pmlmodeltrainingclasse = pmlmodeltraining$classe
pmlmodeltestingclasse = pmlmodeltesting$classe

## Clearing variables with near zero variance as bad predictors
pmlmodeltraining = pmlmodeltraining[, -nearZeroVar(pmlmodeltraining)]
## Clearing variables with more than 95% of NA values
pmlmodeltraining = pmlmodeltraining[, ((colSums(is.na(pmlmodeltraining))/nrow(pmlmodeltraining)) <= 0.95)]

## Removing non relevant columns, and outcome
pmlmodeltraining = pmlmodeltraining[, -c(1:6, 59)]
## Leaving only columns present in the pmlmodel training data set into the testing and validation set
pmlmodeltesting = pmlmodeltesting[, names(pmlmodeltraining)]
pmltesting = pmltesting[, names(pmlmodeltraining)]
## Converting all predictors to numeric
pmlmodeltraining = data.frame(sapply(pmlmodeltraining, as.numeric))
pmlmodeltesting = data.frame(sapply(pmlmodeltesting, as.numeric))
pmltesting = data.frame(sapply(pmltesting, as.numeric))

## Cross validation settings : 10 folds repeat 3 times
control = trainControl(method = 'repeatedcv', number = 10, repeats = 3)
## Fitting the different proposed models
fitTreeCARET = train(classe ~ ., data = cbind(classe = pmlmodeltrainingclasse, pmlmodeltraining),
                     method = 'rpart', trControl = control) 
fitLDA = train(classe ~ ., data = cbind(classe = pmlmodeltrainingclasse, pmlmodeltraining),
               method = 'lda', trControl = control)
fitNB = train(classe ~ ., data = cbind(classe = pmlmodeltrainingclasse, pmlmodeltraining), 
              method = 'nb', trControl = control)
fitGBM = train(classe ~ ., data = cbind(classe = pmlmodeltrainingclasse, pmlmodeltraining), 
               method = 'gbm', trControl = control, verbose = FALSE)
fitRF = train(classe ~ ., data = cbind(classe = pmlmodeltrainingclasse, pmlmodeltraining), 
              method = 'rf', trControl = control)

## Predicted outcome values per model for the training set
predTrainTreeCARET = predict(fitTreeCARET, newdata = pmlmodeltraining)
predTrainLDA = predict(fitLDA, newdata = pmlmodeltraining)
predTrainNB = predict(fitNB, newdata = pmlmodeltraining)
predTrainGBM = predict(fitGBM, newdata = pmlmodeltraining)
predTrainRF = predict(fitRF, newdata = pmlmodeltraining)

## Calculating confusion Matrix for training values per model
metrics = data.frame()
confTrainTreeCARET = confusionMatrix(predTrainTreeCARET, pmlmodeltrainingclasse)
metrics = rbind(metrics, c('Tree-Caret', 'Training', confTrainTreeCARET$overall))
names(metrics) = c('Model', 'Type', names(confTrainTreeCARET$overall))
confTrainLDA = confusionMatrix(predTrainLDA, pmlmodeltrainingclasse)
metrics = rbind(metrics, c('LDA', 'Training', confTrainLDA$overall))
confTrainNB = confusionMatrix(predTrainNB, pmlmodeltrainingclasse)
metrics = rbind(metrics, c('Naive Bayes', 'Training', confTrainNB$overall))
confTrainGBM = confusionMatrix(predTrainGBM, pmlmodeltrainingclasse)
metrics = rbind(metrics, c('GBM', 'Training', confTrainGBM$overall))
confTrainRF = confusionMatrix(predTrainRF, pmlmodeltrainingclasse)
metrics = rbind(metrics, c('Random Forests', 'Training', confTrainRF$overall))

## Predicted outcome values per model for the testing set
predTestTreeCARET = predict(fitTreeCARET, newdata = pmlmodeltesting)
predTestLDA = predict(fitLDA, newdata = pmlmodeltesting)
predTestNB = predict(fitNB, newdata = pmlmodeltesting)
predTestGBM = predict(fitGBM, newdata = pmlmodeltesting)
predTestRF = predict(fitRF, newdata = pmlmodeltesting)

## Calculating confusion Matrix for testing values per model
confTestTreeCARET = confusionMatrix(predTestTreeCARET, pmlmodeltestingclasse)
metrics = rbind(metrics, c('Tree-Caret', 'Testing', confTestTreeCARET$overall))
confTestLDA = confusionMatrix(predTestLDA, pmlmodeltestingclasse)
metrics = rbind(metrics, c('LDA', 'Testing', confTestLDA$overall))
confTestNB = confusionMatrix(predTestNB, pmlmodeltestingclasse)
metrics = rbind(metrics, c('Naive Bayes', 'Testing', confTestNB$overall))
confTestGBM = confusionMatrix(predTestGBM, pmlmodeltestingclasse)
metrics = rbind(metrics, c('GBM', 'Testing', confTestGBM$overall))
confTestRF = confusionMatrix(predTestRF, pmlmodeltestingclasse)
metrics = rbind(metrics, c(Model = 'Random Forests', Type = 'Testing', confTestRF$overall))

print(metrics[, c(1:6)])

## Predicted outcome values per model for the validation pmltesting set
predValidationtRF = predict(fitRF, newdata = pmltesting)
print(predValidationtRF)

## Decision tree using caret train function
print(fitTreeCARET)
## LDA
print(fitLDA)
## Naive Bayes
print(fitNB)
## GBM
print(fitGBM)
## Random Forests
print(fitRF)

## Random Forests
plot(fitRF)
plot(fitRF$finalModel)
## GBM
plot(fitGBM)
plot(fitGBM$finalModel)
