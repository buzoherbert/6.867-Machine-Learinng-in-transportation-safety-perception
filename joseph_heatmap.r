install.packages("NISTunits", dependencies = TRUE);
install.packages("reshape2", dependencies = TRUE);
install.packages("plotly", dependencies = TRUE);

library(plotly);
library(NISTunits);
library(reshape2);

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
  hour = safety_data[["hour"]],
  week_day = safety_data[["wday"]]
);

field1 = "gender"
field2 = "age"

casting_formula = sprintf("%s ~ %s", field1, field2)

#agg = aggregate(safety_data[["pointsecurity"]], list(Gender = safety_data[["gender"]], Age = safety_data[["age"]]), mean, na.rm=TRUE);
agg = dcast(safety_data, casting_formula, mean, value.var = "pointsecurity");
#if("Var.2" %in% colnames(agg))
#{
#  agg = subset(agg, select = -c(Var.2) );
#}
agg = subset(agg, select = -c(Var.2) );
#agg = na.omit(agg);
row.names(agg) <- agg[,1];
agg <- agg[,-1];
#agg[agg > 4] <- 4

num_surveys = dcast(safety_data, casting_formula, length, value.var = "pointsecurity");
#if("Var.2" %in% colnames(agg))
#{
#  num_surveys = subset(num_surveys, select = -c(Var.2) );
#}
num_surveys = subset(num_surveys, select = -c(Var.2) );
num_surveys = na.omit(num_surveys);
#row.names(num_surveys) <- num_surveys[,1];
#num_surveys <- num_surveys[,-1];
num_surveys;

num_list = aggregate(safety_data[["pointsecurity"]], list(field1 = safety_data[[field1]], field2 = safety_data[[field2]]), length);
num_list[num_list==""] <- NA
num_list = num_list[!is.na(num_list$field2),];
num_list;

xa = list(title = field2)
ya = list(title = field1)
anno_x = num_list[["field2"]]
anno_y = num_list[["field1"]]
anno_text = num_list[["x"]]
#anno_x = c("BRT")
#anno_y = c(3)
#anno_text = c("xxx")
plot_ly(
  x = names(agg), y = row.names(agg),
  z = data.matrix(agg), type = "heatmap", colors = colorRamp(c("red", "green"))) %>%
  layout(xaxis = xa, yaxis = ya) %>%
  layout(margin = list(l = 100)) %>%
  add_annotations(x = anno_x, y = anno_y, text = anno_text, xref = 'x', yref = 'y', showarrow = FALSE, font=list(color='black'));
#agg_heat <- heatmap(as.matrix(sapply(agg, as.numeric)), Rowv=NA, Colv=NA, scale='none');
agg;