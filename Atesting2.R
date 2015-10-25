# clean the training data

#read training data and replace "#DIV/0!" with "NA"
traingdata <- read.csv("pml-training.csv", na.strings=c("#DIV/0!","NA"))

#remove columns with no data (all NA)
#traingdata <- traingdata[,colSums(is.na(traingdata))>nrow(traingdata)/10]
trainnacols <- traingdata[,colSums(is.na(traingdata))>nrow(traingdata)/10]

#find index of columns with high counts of NA's
naIndexs <- which(names(traingdata) %in% names(trainnacols))

#I'm also going to remove unused columns (X, user_name, time stamps and new window: column indexs: 1:6)
naIndexs <- append(1:6, naIndexs)

#now remove these columns from the dataset
cleandata <- traingdata[,-naIndexs]

#now only keep lighly variable data (roll, pitch, yaw) and the label classe
savenames <- c("roll_belt","pitch_belt","yaw_belt","roll_arm","pitch_arm","yaw_arm","roll_forearm","pitch_forearm","yaw_forearm","classe")
useIndexs <- which(names(cleandata) %in% savenames)
cleandata2 <- cleandata[,useIndexs]

savenames <- c("num_window","roll_belt","pitch_belt","yaw_belt","roll_arm","pitch_arm","yaw_arm","roll_forearm","pitch_forearm","yaw_forearm","classe")
useIndexs <- which(names(cleandata) %in% savenames)
cleandata3 <- cleandata[,useIndexs]

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
nzv

#also split the reduced data as well for futher testing

library(caret)

set.seed(12345)
inTrain2 <- createDataPartition(y=cleandata2$classe, p=0.6, list=FALSE)

cleantraining2 <- cleandata2[inTrain2,]
cleantesting2 <- cleandata2[-inTrain2,]

dim(cleantraining2)
dim(cleantesting2)

set.seed(12345)
inTrain3 <- createDataPartition(y=cleandata3$classe, p=0.6, list=FALSE)

cleantraining3 <- cleandata3[inTrain3,]
cleantesting3 <- cleandata3[-inTrain3,]

dim(cleantraining3)
dim(cleantesting3)

###################################
# now try some various training attempts:
if (require("doParallel") == FALSE)
{
  install.packages("doParallel")
  library(doParallel)
}
#cl <- makeCluster(10) #setup run to use 10 cores
registerDoParallel(cores = 6)

set.seed(5436)
#does not work (fails to produce modelFit for cleantraing, cleantraing2, or cleantraing3)
#modelFit <- train(classe ~.,data=cleantraining3, method="glm")
#modelFit <- train(classe ~.,data=cleantraining3, preProcess=c("center","scale"), method="glm")
#modelFit <- train(classe ~.,data=cleantraining3, preProcess=c("pca"), method="glm")
#modelFit <- train(classe ~ .,data=cleantraining3,method="lm")

modelFit <- train(classe ~.,data=cleantraining, method="gbm", verbose=FALSE)
modelFit

modelFit <- train(classe ~.,data=cleantraining2, method="gbm", verbose=FALSE)
modelFit

set.seed(5436)
system.time(
  modelFitGBM <- train(classe ~.,data=cleantraining3, method="gbm", verbose=FALSE) #winner at 99.4% in time 95.5 seconds to train
)
modelFitGBM
saveRDS(modelFitGBM,"modelFitGBM.rds")

#try to tune it a bit
ctrl <- trainControl(method = "repeatedcv", repeats = 5)
system.time(
  modelFit2 <- train(classe ~.,data=cleantraining3, method="gbm", verbose=FALSE) 
)
modelFit2


#load my rf model used so far
modFit2b <- readRDS("rfFullModel2b.rds")

#load testing data
testingdata <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!","NA"))

cleantestingdata <- testingdata[,-naIndexs]

answers <- predict(modelFit,newdata=testingdata)

answers2 <- predict(modFit2b,newdata=testingdata)

answers == answers2
answers
