
size <- 100
Pill <-rnorm(size)
Y <- rnorm(size)
Out <- -0.8*Y + rnorm(size)
X <- 0.7*Pill + 0.83*Y + rnorm(size)
C <- 0.94*X + rnorm(size)
mymodel1 <- glm(Out ~ Pill)
summary(mymodel1)

mymodel2 <- glm(Out ~ Pill + X)
summary(mymodel2)

mymodel3 <- glm(Out ~ Pill + C)
summary(mymodel3)


