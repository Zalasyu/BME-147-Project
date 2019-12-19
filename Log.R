# Title     : TODO
# Objective : TODO
# Created by: alec_
# Created on: 12/6/2019

# Title     : TODO
# Objective : TODO
# Created by: alec_
# Created on: 12/6/2019
turing <- matrix(c(12,16,11,15),ncol=2, byrow=TRUE)
colnames(turing) <- c("Pass", "Fail")
rownames(turing) <- c("Female", "Male")

turing <- as.table(turing)
turing

count <- cbind(turing[,2],turing[,1])
count