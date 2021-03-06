install.packages("NISTunits", dependencies = TRUE)
install.packages("lubridate", dependencies = TRUE)

# For working with geographical units
library(NISTunits)
# For parsing dates
library(lubridate)

# loading file
safety_data <- read.table(file="safety_data.csv", header = TRUE, na.strings=c("", "NA"), sep=",")

# Removing rows with no safety perception measurement
completeFun <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])
  return(data[completeVec, ])
}

safety_data = completeFun(safety_data, "pointsecurity")

# Based on
# https://stackoverflow.com/a/365853/3128369
haversine <- function(data, lat1, lon1, lat2, lon2){
  earthRadiusKm = 6371
  
  dLat = NISTdegTOradian(data[[lat2]]-data[[lat1]])
  dLon = NISTdegTOradian(data[[lon2]]-data[[lon1]])
  
  lat1 = NISTdegTOradian(data[[lat1]])
  lat2 = NISTdegTOradian(data[[lat2]])
  
  a = sin(dLat/2) * sin(dLat/2) +
    sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2) 
  c = 2 * atan2(sqrt(a), sqrt(1-a)) 
  distance = earthRadiusKm * c
  return (distance)
}

safety_data[["haversine"]] = haversine(safety_data, "cetram_lat", "cetram_long", "latitude", "longitude")


# Get day of the week
times = strptime(safety_data$date, "%m/%d/%Y")
safety_data$wday = wday(as.Date(times), label=FALSE)

# Making a basic plot of some potentially relevant variables

clean_data <- data.frame(
  # Type of survey
  inside_or_outside = safety_data[["inside_or_outside"]],
  
  # Sociodemographic data
  gender = safety_data[["gender"]],
  age = safety_data[["age"]],
  education = safety_data[["educational_attainment"]],
  
  # Personal trip related data
  origin = safety_data[["origin"]],
  destination = safety_data[["destinations"]],
  companions = safety_data[["companions"]],  
  trip_purpose = safety_data[["trip_purpose"]],
  
  # Perception data
  mode_security = safety_data[["modesecurity"]],
  point_security = safety_data[["pointsecurity"]],
  importance_safety = safety_data[["Importance_safety_digit"]],
  most_safe = safety_data[["mostsafe"]],
  least_safe = safety_data[["leastsafe"]],
  
  # Context data
  bus_or_ped = safety_data[["bus_or_ped"]],
  base_study_zone = safety_data[["base_study_zone"]],
  busdestination = safety_data[["busdestination"]],  
  total_seats = safety_data[["totalseats"]],
  
  
  # Time related contextual information
  haversine = safety_data[["haversine"]],
  urban_typology = safety_data[["urban.typology"]],
  total_passenger_count = safety_data[["totalpassengercount"]],
  total_female_count = safety_data[["totalfemalecount"]],
  empty_seats = safety_data[["emptyseats"]],
  hour = safety_data[["hour"]],
  week_day = safety_data[["wday"]],
  
  # Exact positional information
  latitude = safety_data[["latitude"]],
  longitude = safety_data[["longitude"]]
)

# Treating all the variables as categorical
for(i in names(clean_data)){
  clean_data[[i]] <- as.factor(clean_data[[i]])
}

# Some fields have to be treated as numeric.
# TODO improve this so we don't convert the data twice.
clean_data[["total_female_count"]] = as.numeric(as.character(clean_data[["total_female_count"]]))
clean_data[["total_passenger_count"]] = as.numeric(as.character(clean_data[["total_passenger_count"]]))
clean_data[["empty_seats"]] = as.numeric(as.character(clean_data[["empty_seats"]]))
clean_data[["point_security"]] = as.numeric(as.character(clean_data[["point_security"]]))
clean_data[["haversine"]] = as.numeric(as.character(clean_data[["haversine"]]))
clean_data[["latitude"]] = as.numeric(as.character(clean_data[["latitude"]]))
clean_data[["longitude"]] = as.numeric(as.character(clean_data[["longitude"]]))

#Removing incomplete cases
clean_data = na.omit(clean_data)

#Writing to a file
write.csv(clean_data, "safety_data_clean_latlon.csv")
