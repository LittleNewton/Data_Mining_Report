# 这是R语言的联系文档
###

#z <- 4

# 使用c()函数创建一个向量
c(4,7,23.5,76.2,80) -> v

length(v)

# 查看v的类型
mode(v)

# R的强制类型转换
c(4,7,23.5,76.2,80,'rrt')
v

# NA可以在任何的向量中出现，代表是空值
c(4,6,NA,2) -> u

# R语言的布尔值
c(T,F,NA,TRUE,T)

# 

# R允许创建空向量
vector() -> x
# 貌似R也是不会检查下标越界的

# R的函数是向量化的
c(4,7,23.4,76.2,80) -> v
sqrt(v) -> x
x

# R的向量加法比较特别，不等长的向量也可以
c(4,6,8,24) -> v1
c(10,2) -> v2
v1 + v2
# 这里采用的是短的循环补位知道登长再加

# 因子提供了一个简单紧凑的形式来处理分类（名义）数据
# 因子用levels来表示所有可能的取值
c('f','m','m','m','f','m','f','m','f','f') -> g
g <- factor(g)
# f不再是一个字符向量，而是一个数值向量，R内部就是用数值存储的

# R是函数式编程语言，最常见的应用函数的方式之一就是函数复合
factor(c('m','m','m','m'),levels = c('f','m')) -> other.g
# 这里把factor()应用到了c()上

# 计数因子数目用table()函数
table(other.g)
table(f)

# 
a <- factor(c('adult','adult','juvenile','juvenile','adult','adult','adult','juvenile', 'adult','juvenile'))
table(a,g)