#project initial try

#from http://groupware.les.inf.puc-rio.br/har
#
#Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
#   exactly according to the specification (Class A), 
#   throwing the elbows to the front (Class B), 
#   lifting the dumbbell only halfway (Class C), 
#   lowering the dumbbell only halfway (Class D) and 
#   throwing the hips to the front (Class E).

#Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. 
#Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. 
#The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. 
#We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

#class project site
#
#The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 
#You may use any of the other variables to predict with. You should create a report describing how you built your model, 
#how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. 
#You will also use your prediction model to predict 20 different test cases. 
#
#1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. 
#   Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. 
#   It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
#
#2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. 
#   Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 

#################################
# clean the training data

#read training data and replace "#DIV/0!" with "NA"
traingdata <- read.csv("pml-training.csv", na.strings=c("#DIV/0!","NA"))

#remove columns with no data (all NA)
#traingdata <- traingdata[,colSums(is.na(traingdata))>nrow(traingdata)/10]
trainnacols <- traingdata[,colSums(is.na(traingdata))>nrow(traingdata)/10]

#find index of columns with high counts of NA's
naIndexs <- which(names(traingdata) %in% names(trainnacols))

#I'm also going to remove unused columns (X, user_name, time stamps and window: column indexs: 1:7)
naIndexs <- append(1:7, naIndexs)

#now remove these columns from the dataset
cleandata <- traingdata[,-naIndexs]

#################################
# split training data to allow training and testing

library(caret)

set.seed(12345)
inTrain <- createDataPartition(y=cleandata$classe, p=0.6, list=FALSE)

cleantraining <- cleandata[inTrain,]
cleantesting <- cleandata[-inTrain,]

dim(cleantraining)
dim(cleantesting)

##############
#try a simple glm model first

set.seed(5436)
#does not work (fails to produce modelFit)
#modelFit <- train(classe ~.,data=cleantraining, method="glm")
#modelFit <- train(classe ~.,data=cleantraining, preProcess=c("center","scale"), method="glm")
#modelFit <- train(classe ~.,data=cleantraining, preProcess=c("pca"), method="glm")
#modelFit <- train(classe ~ .,data=cleantraining,method="lm")

modelFit <- train(classe ~.,data=cleantraining, method="gbm", verbose=FALSE)
modelFit$finalModel
saveRDS(modelFit,"gbmFullModel.rds")

modelFit <- train(classe ~ .,data=cleantraining,method="rpart")
modelFit$finalModel

#doMC is not available for R version 3.2.2
#if (require("doMC") == FALSE)
#{
#  install.packages("doMC")
#  library(doMC)
#}
#registerDoMC(cores = 10)

if (require("foreach") == FALSE)
{
  install.packages("foreach")
  library(foreach)
}
if (require("doParallel") == FALSE)
{
  install.packages("doParallel")
  library(doParallel)
}
cl <- makeCluster(10) #setup run to use 10 cores
registerDoParallel(cl)

## All subsequent models are then run in parallel
modFit <- train(classe ~ .,data=cleantraining,method="rf",prox=TRUE)
modFit
saveRDS(modFit,"rfFullModel.rds")

modFit <- readRDS("rfFullModel.rds")

#Getting a single tree

getTree(modFit$finalModel,k=2)

pred <- predict(modFit,cleantesting)

#figure prediction success
sum(pred == cleantesting$classe)/length(pred) #comes out to  99.27352%

##############
#next try 

#svm
modsvmFit <- train(classe ~ ., data=cleantraining, method="svmRadialWeights")
modsvmFit

saveRDS(modsvmFit,"svmRadialWeightsModel.rds")

############################
#try parallel processing
if (require("doParallel") == FALSE)
{
  install.packages("doParallel")
  library(doParallel)
}
#cl <- makeCluster(10) #setup run to use 10 cores
registerDoParallel(cores = 6)

#next try rotationForest
#modrotfFit <- train(classe ~ ., data=cleantraining, method="rotationForest")
#modrotfFit

#saveRDS(modrotfFit,"rotationForestModel.rds")

#next try penalized multinomial regression (multinom)
modmultnFit <- train(classe ~ ., data=cleantraining, method="multinom")
modmultnFit

saveRDS(modmultnFit,"multinomModel.rds")

#next try k nearest neighbors (knn)
modknnFit <- train(classe ~ ., data=cleantraining, method="knn")
modknnFit
saveRDS(modknnFit,"knnModel.rds")

#next try linear discrminant analysis (lda)
modldaFit <- train(classe ~ ., data=cleantraining, method="lda")
modldaFit
saveRDS(modldaFit,"ldaModel.rds")

#next try bagged logistic regression (logicBag)
#modlogbagFit <- train(classe ~ ., data=cleantraining, method="logicBag")
#modlogbagFit
#saveRDS(modlogbagFit,"logicBagModel.rds")

################################################
#experiment with various rf configuration to view and analyze
if (require("doParallel") == FALSE)
{
  install.packages("doParallel")
  library(doParallel)
}
#cl <- makeCluster(10) #setup run to use 10 cores
registerDoParallel(cores = 6)

#standard default train setup
set.seed(4532)
modFit <- train(classe ~ .,data=cleantraining,method="rf",prox=TRUE)
modFit
saveRDS(modFit,"rfFullModel.rds")

#change to repeated cv with repeats of 5
set.seed(4532)
ctrl <- trainControl(method = "repeatedcv", repeats = 5)

modFit2 <- train(classe ~ ., data = cleantraining, method = "rf",prox=TRUE, trControl = ctrl)
modFit2
saveRDS(modFit2,"rfFullModel2.rds")

#change performance metrics (note, twoclass summary is not appropirate for 5 outputs)
set.seed(4532)
ctrl <- trainControl(method="repeatedcv", repeats=6, classProbs=TRUE)

modFit3 <- train(classe ~ ., data = cleantraining, method = "rf",prox=TRUE, trControl = ctrl)
modFit3
saveRDS(modFit3,"rfFullModel3.rds")

