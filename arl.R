dataset <- read.csv('Market_Basket_Optimisation.csv', header = FALSE)
summary(dataset)
library(arules)

dataset <- read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
# install.packages('arules')

library(arules)
itemFrequencyPlot(dataset, topN = 10)
rules <- apriori(dataset, parameter = list(support = 0.003, confidence = 0.4))
inspect(sort(rules, by = "lift")[1:20])
