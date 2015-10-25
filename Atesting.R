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

nzv <- nearZeroVar(cleantraining,saveMetrics=TRUE)
sumnzv = sum(nzv$nzv)

names(cleantraining)
#featurePlot(x=cleantraining[,-53],y = cleantraining$classe,plot="pairs")

###
#load the model
modFit <- readRDS("rfFullModel2b.rds")

#confustion matrix for final model
modFit$finalModel$confusion

#load verification set
predictions <- predict(modFit,newdata=cleantesting)
predictions

#figure prediction success
sum(predictions == cleantesting$classe)/length(predictions) #comes out to  99.68137%

#calcualte prediction accuracy

testingdata <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!","NA"))

answers2 <- predict(modFit,newdata=testingdata)
answers2
#sum(answers2 == testingdata$classe)/length(answers2) #comes out to  99.20979%

answers == answers2 #all the same (whew)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
