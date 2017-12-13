install.packages("plyr")
library(plyr)

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