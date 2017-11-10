install.packages("NISTunits", dependencies = TRUE);
install.packages("GGally", dependencies = TRUE);
install.packages("ggplot2", dependencies = TRUE);


library(NISTunits);
# Graphing library
 library(GGally);
 library(ggplot2);
# For parsing dates
library(lubridate);

# loading file
safety_data = read.csv("safety_data.csv");

# Removing rows with no safety perception measurement
completeFun <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])
  return(data[completeVec, ])
}

safety_data = completeFun(safety_data, "pointsecurity")

# Based on
# https://stackoverflow.com/a/365853/3128369
haversine <- function(data, lat1, lon1, lat2, lon2){
  earthRadiusKm = 6371;
  
  dLat = NISTdegTOradian(data[[lat2]]-data[[lat1]]);
  dLon = NISTdegTOradian(data[[lon2]]-data[[lon1]]);
  
  lat1 = NISTdegTOradian(data[[lat1]]);
  lat2 = NISTdegTOradian(data[[lat2]]);
  
  a = sin(dLat/2) * sin(dLat/2) +
    sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2); 
  c = 2 * atan2(sqrt(a), sqrt(1-a)); 
  distance = earthRadiusKm * c;
  return (distance);
}

safety_data[["haversine"]] = haversine(safety_data, "cetram_lat", "cetram_long", "latitude", "longitude")

# get hours

times = strptime(safety_data$time, "%I:%M:%S %p");

safety_data$hour = as.factor(hour(times));
safety_data = subset(safety_data, select = -c(time) )

# Making a basic plot of some potentially relevant variables

plot_data <- data.frame(
  bus_or_ped = safety_data[["bus_or_ped"]],
  base_study_zone = safety_data[["base_study_zone"]],
  busdestination = safety_data[["busdestination"]],
  inside_or_outside = safety_data[["inside_or_outside"]],
  total_seats = safety_data[["totalseats"]],
  total_passenger_count = safety_data[["totalpassengercount"]],
  total_female_count = safety_data[["totalfemalecount"]],
  empty_seats = safety_data[["emptyseats"]],
  gender = safety_data[["gender"]],
  age = safety_data[["age"]],
  companions = safety_data[["companions"]],  
  education = safety_data[["educational_attainment"]],
  origin = safety_data[["origin"]],
  destination = safety_data[["destinations"]],
  trip_purpose = safety_data[["trip_purpose"]],
  mode_security = safety_data[["modesecurity"]],
  point_security = safety_data[["pointsecurity"]],
  importance_safety = safety_data[["Importance_safety_digit"]],
  most_safe = safety_data[["mostsafe"]],
  least_safe = safety_data[["leastsafe"]],
  urban_typology = safety_data[["urban.typology"]],
  haversine = safety_data[["haversine"]],
  hour = safety_data[["hour"]]
);

# Treating all the variables as categorical
for(i in names(plot_data)){
  plot_data[[i]] <- as.factor(plot_data[[i]])
}

plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]))


# Getting a summary of the data
# summary(plot_data)

# plotting the data
to_plot = plot_data;
to_plot$point_security = as.factor(to_plot$point_security)
ggpairs(to_plot, mapping = aes(color = point_security))


########################
## Linear regression model

set.seed(888)
smp_size <- floor(0.75*nrow(plot_data))
train_data <- sample(seq_len(nrow(plot_data)), size = smp_size)
train <- plot_data[train_data,]
test <- plot_data[-train_data,]
linear_model = lm(train$point_security~., data = train)
summary(linear_model)

linear_model2 = lm(train$point_security~ +haversine +gender +companions +education +base_study_zone +busdestination +inside_or_outside, data = train)
summary(linear_model2)

pred_data = predict(linear_model2, newdata = test)
SSE = sum((pred_data - test$point_security)^2)
pred_mean = mean(train$point_security)
SST = sum((pred_mean - test$point_security)^2)
OSR2 = 1-SSE/SST
RMSE = sqrt(sum((pred_data - test$point_security)^2)/nrow(test))
MAE = sum(abs(pred_data - test$point_security))/nrow(test)