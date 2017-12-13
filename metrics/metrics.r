install.packages("plyr")
install.packages("ggplot2")

library(plyr)
library(ggplot2)

metrics = NULL

file_list <- list.files()

for (file in file_list){
  if(grepl(".csv", file)){
    # if the merged dataset doesn't exist, create it
    if (!exists("metrics")){
      metrics <- read.table(file, header=TRUE, sep=",")
      metrics$model = gsub(".csv", "", file)
    }
    
    # if the merged dataset does exist, append to it
    if (exists("metrics")){
      temp_dataset <-read.table(file, header=TRUE, sep=",")
      temp_dataset$model = gsub(".csv", "", file)
      metrics<-rbind.fill(metrics, temp_dataset)
      rm(temp_dataset)
    }
  }
}



metrics$category = revalue(metrics$category, c("Trip context"="Contextual", "Personal trip"="Trip", "Time context" = "Instant", "All variables" = "All"))



variables = list("osr2", "accuracy", "sse", "f1_score")


p <- ggplot(metrics, aes(category, osr2, label = model))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/cat_osr2.png")
p <- ggplot(metrics, aes(category, accuracy, label = model))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/cat_acc.png")
p <- ggplot(metrics, aes(category, sse, label = model))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/cat_sse.png")
p <- ggplot(metrics, aes(category, f1_score, label = model))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/cat_f1.png")

p <- ggplot(metrics, aes(model, osr2, label = category))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/mod_osr2.png")
p <- ggplot(metrics, aes(model, accuracy, label =category))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/mod_acc.png")
p <- ggplot(metrics, aes(model, sse, label = category))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/mod_sse.png")
p <- ggplot(metrics, aes(model, f1_score, label = category))
p + geom_boxplot()  + geom_point() + geom_text(hjust = 0, nudge_x = 0.05, check_overlap = FALSE, size = 3, angle = 30)+ theme(axis.text.x = element_text(size=10, angle=30))
ggsave("./graphs/mod_f1.png")
