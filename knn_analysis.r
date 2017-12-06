# For knn model
library("class");

# KNN model
new_safety_data <- read.table(file="final_data_0.csv", header = TRUE, na.strings=c("", "NA"), sep=",")

for(i in names(new_safety_data)){
  new_safety_data[[i]] <- as.numeric(new_safety_data[[i]])
}

# Use first 60% of data for training
new_train_ind <- nrow(new_safety_data)*0.6
new_val_ind <- new_train_ind + nrow(new_safety_data)*0.2

new_train <- new_safety_data[1:new_train_ind-1,]
new_val <- new_safety_data[new_train_ind:new_val_ind-1,]
new_test <- new_safety_data[new_val_ind:nrow(new_safety_data),]

# Finding distribution of point_security in new_test
point_distribution = as.data.frame(table(new_val$point_security))
for(i in 1:5){
  point_distribution[i, "Freq_Dec"] <- point_distribution[i, "Freq"]/nrow(new_val)
}
point_distribution
#point_distribution
#Var1 Freq  Freq_Dec
#1    1   40 0.2162162
#2    2   26 0.1405405
#3    3   58 0.3135135
#4    4   40 0.2162162
#5    5   21 0.1135135

find_simi <- function(prop_table) {
  accuracy <- 0
  for(i in 1:5){
    accuracy <- accuracy + prop_table[i, i] 
  }
  return (accuracy)
}

nn9 <- knn (new_train, new_val, new_train$point_security, k=9)
table(nn9, new_val$point_security)
prop_table <- prop.table(table(nn9, new_val$point_security))
find_simi(prop_table)
# accuracy = 0.3243243

nn11 <- knn (new_train, new_val, new_train$point_security, k=11)
table(nn11, new_val$point_security)
prop_table <- prop.table(table(nn11, new_val$point_security))
find_simi(prop_table)
# accuracy = 0.3405405

nn13 <- knn (new_train, new_val, new_train$point_security, k=13)
table(nn13, new_val$point_security)
prop_table <- prop.table(table(nn13, new_val$point_security))
find_simi(prop_table)
# accuracy = 0.3297297

# without numerical values
# plot_data[["total_female_count"]] = as.numeric(as.character(plot_data[["total_female_count"]]));
# plot_data[["total_passenger_count"]] = as.numeric(as.character(plot_data[["total_passenger_count"]]));
# plot_data[["empty_seats"]] = as.numeric(as.character(plot_data[["empty_seats"]]));
# plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]));
# plot_data[["haversine"]] = as.numeric(plot_data[["haversine"]]);

new_train <- subset(new_train, select=-haversine-total_female_count-total_passenger_count-empty_seats)
new_test <- subset(new_test, select=-haversine-total_female_count-total_passenger_count-empty_seats)

nn3 <- knn (new_train, new_test, new_train$point_security, k=3)
table(nn3, new_test$point_security)
prop.table(table(nn3, new_test$point_security))

nn5 <- knn (new_train, new_test, new_train$point_security, k=5)
table(nn5, new_test$point_security)
prop.table(table(nn5, new_test$point_security))

# still the best
nn7 <- knn (new_train, new_test, new_train$point_security, k=7)
table(nn7, new_test$point_security)
prop.table(table(nn7, new_test$point_security))

nn9 <- knn (new_train, new_test, new_train$point_security, k=9)
table(nn9, new_test$point_security)
prop.table(table(nn9, new_test$point_security))

nn11 <- knn (new_train, new_test, new_train$point_security, k=11)
table(nn11, new_test$point_security)
prop.table(table(nn11, new_test$point_security))