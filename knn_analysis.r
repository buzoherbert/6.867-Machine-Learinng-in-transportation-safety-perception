# For knn model
library("class");

set.seed(888)

# KNN model
new_safety_data <- read.table(file="final_data_0.csv", header = TRUE, na.strings=c("", "NA"), sep=",")

for(i in names(new_safety_data)){
  new_safety_data[[i]] <- as.numeric(new_safety_data[[i]])
}

# Use first 60% of data for training
new_train_ind <- floor(nrow(new_safety_data)*0.6)
new_val_ind <- new_train_ind + floor(nrow(new_safety_data)*0.2)

new_train <- new_safety_data[1:(new_train_ind-1),]
new_val <- new_safety_data[(new_train_ind):(new_val_ind),]
new_test <- new_safety_data[(new_val_ind+1):nrow(new_safety_data),]

train_use <- subset(new_train, select=-point_security)
val_use <- subset(new_val, select=-point_security)
test_use <- subset(new_test, select=-point_security)

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

f1_score <- function(predicted, expected, positive.class="5") {
  predicted <- as.factor(predicted)
  expected  <- as.factor(expected)
  cm = as.matrix(table(expected, predicted))
  
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <-  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  
  #Assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0
  
  #Binary F1 or Multi-class macro-averaged F1
  ifelse(nlevels(expected) == 2, f1[positive.class], mean(f1))
}

nn9 <- knn (train_use, val_use, new_train$point_security, k=9)
table(nn9, new_val$point_security)
prop_table <- prop.table(table(nn9, new_val$point_security))
find_simi(prop_table)
# accuracy = 0.2864865
f1_score(nn9, new_val$point_security)
# f1score = 0.2348394

nn11 <- knn (train_use, val_use, new_train$point_security, k=11)
table(nn11, new_val$point_security)
prop_table <- prop.table(table(nn11, new_val$point_security))
find_simi(prop_table)
# accuracy = 0.3081081
f1_score(nn11, new_val$point_security)
# f1score = 0.2578481

nn13 <- knn (train_use, val_use, new_train$point_security, k=13)
table(nn13, new_val$point_security)
prop_table <- prop.table(table(nn13, new_val$point_security))
find_simi(prop_table)
# accuracy = 0.3135135
f1_score(nn13, new_val$point_security)
# f1score = 0.2456826

# nn11 seems to be the best
nn11 <- knn (train_use, test_use, new_train$point_security, k=11)
table(nn11, new_test$point_security)
prop_table <- prop.table(table(nn11, new_test$point_security))
find_simi(prop_table)
# accuracy = 0.2648649
f1_score(nn11, new_test$point_security)
# f1score = 0.2097597

#************************************************************************************

# Sociodemographic Data
nnew_train <- data.frame(
  point_security = new_train[["point_security"]],
  gender = new_train[["gender"]],
  age = new_train[["age"]],
  education = new_train[["education"]]
)

nnew_val <- data.frame(
  point_security = new_val[["point_security"]],
  gender = new_val[["gender"]],
  age = new_val[["age"]],
  education = new_val[["education"]]
)
nnew_test <- data.frame(
  point_security = new_test[["point_security"]],
  gender = new_test[["gender"]],
  age = new_test[["age"]],
  education = new_test[["education"]]
)

train_usee <- subset(nnew_train, select=-point_security)
val_usee <- subset(nnew_val, select=-point_security)
test_usee <- subset(nnew_test, select=-point_security)

nn2 <- knn (train_usee, val_usee, nnew_train$point_security, k=2)
table(nn2, nnew_val$point_security)
prop_table <- prop.table(table(nn2, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.2864865
f1_score(nn2, nnew_val$point_security)
# f1score = 0.1427113

nn3 <- knn (train_usee, val_usee, nnew_train$point_security, k=3)
table(nn3, nnew_val$point_security)
prop_table <- prop.table(table(nn3, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.2972973
f1_score(nn3, nnew_val$point_security)
# f1score = 0.1595887

nn4 <- knn (train_usee, val_usee, nnew_train$point_security, k=4)
table(nn4, nnew_val$point_security)
prop_table <- prop.table(table(nn4, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.2864865
f1_score(nn4, nnew_val$point_security)
# f1score = 0.1435102

#************************************************************************************

# Personal Trip Related Data
nnew_train <- data.frame(
  point_security = new_train[["point_security"]],
  origin = new_train[["origin"]],
  destinations = new_train[["destination"]],
  companions = new_train[["companions"]],
  trip_purpose = new_train[["trip_purpose"]]
)

nnew_val <- data.frame(
  point_security = new_val[["point_security"]],
  origin = new_val[["origin"]],
  destinations = new_val[["destination"]],
  companions = new_val[["companions"]],
  trip_purpose = new_val[["trip_purpose"]]
)
nnew_test <- data.frame(
  point_security = new_test[["point_security"]],
  origin = new_test[["origin"]],
  destinations = new_test[["destination"]],
  companions = new_test[["companions"]],
  trip_purpose = new_test[["trip_purpose"]]
)

train_usee <- subset(nnew_train, select=-point_security)
val_usee <- subset(nnew_val, select=-point_security)
test_usee <- subset(nnew_test, select=-point_security)

nn2 <- knn (train_usee, val_usee, nnew_train$point_security, k=2)
table(nn2, nnew_val$point_security)
prop_table <- prop.table(table(nn2, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.2972973
f1_score(nn2, nnew_val$point_security)
# f1score = 0.2413235

nn3 <- knn (train_usee, val_usee, nnew_train$point_security, k=3)
table(nn3, nnew_val$point_security)
prop_table <- prop.table(table(nn3, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3297297
f1_score(nn3, nnew_val$point_security)
# f1score = 0.2964836

nn4 <- knn (train_usee, val_usee, nnew_train$point_security, k=4)
table(nn4, nnew_val$point_security)
prop_table <- prop.table(table(nn4, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.2972973
f1_score(nn4, nnew_val$point_security)
# f1score = 0.2472851

#************************************************************************************

# Perception Data
nnew_train <- data.frame(
  point_security = new_train[["point_security"]],
  mode_security = new_train[["mode_security"]],
  importance_safety = new_train[["importance_safety"]],
  most_safe = new_train[["most_safe"]],
  least_safe = new_train[["least_safe"]]
)

nnew_val <- data.frame(
  point_security = new_val[["point_security"]],
  mode_security = new_val[["mode_security"]],
  importance_safety = new_val[["importance_safety"]],
  most_safe = new_val[["most_safe"]],
  least_safe = new_val[["least_safe"]]
)
nnew_test <- data.frame(
  point_security = new_test[["point_security"]],
  mode_security = new_test[["mode_security"]],
  importance_safety = new_test[["importance_safety"]],
  most_safe = new_test[["most_safe"]],
  least_safe = new_test[["least_safe"]]
)

train_usee <- subset(nnew_train, select=-point_security)
val_usee <- subset(nnew_val, select=-point_security)
test_usee <- subset(nnew_test, select=-point_security)

nn3 <- knn (train_usee, val_usee, nnew_train$point_security, k=3)
table(nn3, nnew_val$point_security)
prop_table <- prop.table(table(nn3, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3837838
f1_score(nn3, nnew_val$point_security)
# f1score = 0.3473132

nn4 <- knn (train_usee, val_usee, nnew_train$point_security, k=5)
table(nn4, nnew_val$point_security)
prop_table <- prop.table(table(nn4, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3783784
f1_score(nn4, nnew_val$point_security)
# f1score = 0.3530125

nn5 <- knn (train_usee, val_usee, nnew_train$point_security, k=5)
table(nn5, nnew_val$point_security)
prop_table <- prop.table(table(nn5, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3837838
f1_score(nn5, nnew_val$point_security)
# f1score = 0.3493297

#************************************************************************************

# Context Data
nnew_train <- data.frame(
  point_security = new_train[["point_security"]],
  bus_or_ped = new_train[["bus_or_ped"]],
  base_study_zone = new_train[["base_study_zone"]],
  busdestination = new_train[["busdestination"]],
  total_seats = new_train[["total_seats"]]
)

nnew_val <- data.frame(
  point_security = new_val[["point_security"]],
  bus_or_ped = new_val[["bus_or_ped"]],
  base_study_zone = new_val[["base_study_zone"]],
  busdestination = new_val[["busdestination"]],
  total_seats = new_val[["total_seats"]]
)
nnew_test <- data.frame(
  point_security = new_test[["point_security"]],
  bus_or_ped = new_test[["bus_or_ped"]],
  base_study_zone = new_test[["base_study_zone"]],
  busdestination = new_test[["busdestination"]],
  total_seats = new_test[["total_seats"]]
)

train_usee <- subset(nnew_train, select=-point_security)
val_usee <- subset(nnew_val, select=-point_security)
test_usee <- subset(nnew_test, select=-point_security)

nn4 <- knn (train_usee, val_usee, nnew_train$point_security, k=4)
table(nn4, nnew_val$point_security)
prop_table <- prop.table(table(nn4, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3135135
f1_score(nn4, nnew_val$point_security)
# f1score = 0.2044942

nn5 <- knn (train_usee, val_usee, nnew_train$point_security, k=5)
table(nn5, nnew_val$point_security)
prop_table <- prop.table(table(nn5, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3243243
f1_score(nn5, nnew_val$point_security)
# f1score = 0.2119738

nn6 <- knn (train_usee, val_usee, nnew_train$point_security, k=6)
table(nn6, nnew_val$point_security)
prop_table <- prop.table(table(nn6, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3027027
f1_score(nn6, nnew_val$point_security)
# f1score = 0.1973039

#************************************************************************************

# Time related contextual information
nnew_train <- data.frame(
  point_security = new_train[["point_security"]],
  haversine = new_train[["haversine"]],
  urban_typology = new_train[["urban_typology"]],
  total_passenger_count = new_train[["total_passenger_count"]],
  total_female_count = new_train[["total_female_count"]],
  empty_seats = new_train[["empty_seats"]],
  hour_sin = new_train[["hour_sin"]],
  hour_cos = new_train[["hour_cos"]],
  week_day = new_train[["week_day"]]
)

nnew_val <- data.frame(
  point_security = new_val[["point_security"]],
  haversine = new_val[["haversine"]],
  urban_typology = new_val[["urban_typology"]],
  total_passenger_count = new_val[["total_passenger_count"]],
  total_female_count = new_val[["total_female_count"]],
  empty_seats = new_val[["empty_seats"]],
  hour_sin = new_val[["hour_sin"]],
  hour_cos = new_val[["hour_cos"]],
  week_day = new_val[["week_day"]]
)
nnew_test <- data.frame(
  point_security = new_test[["point_security"]],
  haversine = new_test[["haversine"]],
  urban_typology = new_test[["urban_typology"]],
  total_passenger_count = new_test[["total_passenger_count"]],
  total_female_count = new_test[["total_female_count"]],
  empty_seats = new_test[["empty_seats"]],
  hour_sin = new_test[["hour_sin"]],
  hour_cos = new_test[["hour_cos"]],
  week_day = new_test[["week_day"]]
)

train_usee <- subset(nnew_train, select=-point_security)
val_usee <- subset(nnew_val, select=-point_security)
test_usee <- subset(nnew_test, select=-point_security)

nn6 <- knn (train_usee, val_usee, nnew_train$point_security, k=6)
table(nn6, nnew_val$point_security)
prop_table <- prop.table(table(nn6, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.2864865
f1_score(nn6, nnew_val$point_security)
# f1score = 0.2703044

nn7 <- knn (train_usee, val_usee, nnew_train$point_security, k=7)
table(nn7, nnew_val$point_security)
prop_table <- prop.table(table(nn7, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.3243243
f1_score(nn7, nnew_val$point_security)
# f1score = 0.2955782

nn8 <- knn (train_usee, val_usee, nnew_train$point_security, k=8)
table(nn8, nnew_val$point_security)
prop_table <- prop.table(table(nn8, nnew_val$point_security))
find_simi(prop_table)
# accuracy = 0.2864865
f1_score(nn8, nnew_val$point_security)
# f1score = 0.2586119

