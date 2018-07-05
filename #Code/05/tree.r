library(rpart)
library(rpart.plot)
library(caret)

mydata <- read.csv("Data_Breast_Cancer.CSV")
head(mydata)

set.seed(1000)
train.idx <- createDataPartition(mydata$Class,p=0.7,list=FALSE)

mod <- rpart(Class~.,data=mydata[train.idx,], method="class", control=rpart.control(minsplit=20,cp=0.01))
mod

prp(mod, type=2, extra=104, nn=TRUE, fallen.leaves=TRUE, faclen=4, varlen=3, shadow.col="gray")
mod$cptable

mod.pruned=prune(mod, mod$captable[5, "CP"])

pred.pruned <- predict (mod, mydata[-train, idx, ], type = "class")
table(mydata[-train.idx,]$Class, pred.pruned, dnn=c("Actualn", "Predicted"))