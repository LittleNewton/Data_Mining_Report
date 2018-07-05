library(class)
data <- read.csv("Data_Heart_Disease.CSV")
dim(data)

attributes(data)

data[1:10,]

set.seed(1000)
ind <- sample(2, nrow(data), replace=TRUE, prob - c(0.7, 0.3))
data.train <- data[ind == 1,]
data.test <- data[ind == 2,]
# train a decision tree
library(rpart)
myFoumula <- Chest~Class+Blood + Serun + Sugar + Electrocardiographic + Heartrate + Angina + Oldpeak + Slope + Vessels + Thal
data_rpart <- rpart(myFormula, data=data.train, control=rpart.control(minsplit=10))
attributes(data_rpart)

plot(data_rpart)
text(data_rpart, use.n = T)

opt <- which.min(data_rpart$cptable[, "xerror"])
cp <- data_ra$captable[opt, "CP"]
data_prune <- prune(data_rpart, cp=cp)
print(data_prune)

plot(data_prune)
text(data_prune, use.n = T)

Chest_pred <- predict(data_prune, newdata=data.test)
xlim <- range(data$Chest)
plot(Chest_pred~Chest, data=data.test, xlab="Observed", ylab="Predicted", ylim=xlim, xlim = xlim)
abline(a=0, b=1)