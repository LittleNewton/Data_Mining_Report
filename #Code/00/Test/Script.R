y <- matrix(1:20, nrow = 5, ncol = 4)
y

cells <- c(1, 26, 24, 68)
rnames <- c("R1", "R2")
cnames <- c("C1", "C2")

mymatrix <- matrix(cells, nrow = 2, ncol = 2, byrow = TRUE, dimnames = list(rnames, cnames))
mymatrix
mymatrix2 <- matrix(cells, nrow = 2, ncol = 2, byrow = FALSE, dimnames = list(rnames, cnames))
mymatrix2


x <- matrix(1:10, nrow = 2)
x
x[2,]
x[, 2]
x[1, 4]

dim1 <- c("A1", "A2")
dim2 <- c("B1", "B2", "B3")
dim3 <- c("C1", "C2", "C3", "C4")
z <- array(1:24, c(2, 3, 4), dimnames = list(dim1, dim2, dim3))
z

patientID <- c(1, 2, 3, 4)
age <- c(25, 34, 28, 52)
diabetes <- c("Type1", "Type2", "Type3", "Type4");
status <- c("Poor", "Improved", "Excellent", "Poor")
patientdata <- data.frame(patientID, age, diabetes, status)
patientdata

patientdata[1:2]                        # 取出多少列，还是一个数组
patientdata[c("diabetes", "status")]    # 传一个数组进去，数组是列名的符号串数组
patientdata$age                         # 取出某一列并转置，这与matlab不太一样

summary(mtcars$mpg)
plot(mtcars$mpg, mtcars$disp)
plot(mtcars$mpg, mtcars$wt)

attach(mtcars)
    summary(mpg)
    plot(mpg, disp)
    plot(mpg, wt)
detach(mtcars)
