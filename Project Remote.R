#################################
# clean the training data

#read training data and replace "#DIV/0!" with "NA"
traingdata <- read.csv("pml-training.csv", na.strings=c("#DIV/0!","NA"))

#remove columns with no data (all NA)
trainnacols <- traingdata[,colSums(is.na(traingdata))>nrow(traingdata)/10]

#find index of columns with high counts of NA's
naIndexs <- which(names(traingdata) %in% names(trainnacols))

#I'm also going to remove unused columns (X, user_name, time stamps and window: column indexs: 1:7)
naIndexs <- append(1:7, naIndexs)

#now remove these columns from the dataset
cleandata <- traingdata[,-naIndexs]

if (require("caret") == FALSE)
{
  install.packages("caret", dependencies = c("Depends", "Suggests"))
  library(caret)
}

set.seed(12345)
inTrain <- createDataPartition(y=cleandata$classe, p=0.6, list=FALSE)

cleantraining <- cleandata[inTrain,]
cleantesting <- cleandata[-inTrain,]

set.seed(5436)


system.time(
  modelFit <- train(classe ~ .,data=cleantraining,method="rpart")
)
#13 seconds

modelFit

modelFit$finalModel

################################################
#experiment with various rf configuration to view and analyze
if (require("doParallel") == FALSE)
{
  install.packages("doParallel")
  library(doParallel)
}
#cl <- makeCluster(10) #setup run to use 10 cores
registerDoParallel(cores = 12)

#standard default train setup
set.seed(4532)
system.time(
  modFit <- train(classe ~ .,data=cleantraining,method="rf",prox=TRUE)
)
modFit
saveRDS(modFit,"rfFullModel.rds")
rm(modFit)


#change to repeated cv with repeats of 5
set.seed(4532)
ctrl <- trainControl(method = "repeatedcv", repeats = 5)


system.time(
  modFit2 <- train(classe ~ ., data = cleantraining, method = "rf",prox=TRUE, trControl = ctrl)
)

modFit2
saveRDS(modFit2,"rfFullModel2.rds")
rm(modFit2)

#change performance metrics (note, twoclass summary is not appropirate for 5 outputs)
set.seed(4532)
ctrl <- trainControl(method="repeatedcv", repeats=10, classProbs=TRUE)

system.time(
  modFit3 <- train(classe ~ ., data = cleantraining, method = "rf",prox=TRUE, trControl = ctrl)
)

modFit3
saveRDS(modFit3,"rfFullModel3.rds")
rm(modFit3)