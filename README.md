# NYC TLC Trip Record Data Repository

This repository hosts datasets from the New York City Taxi and Limousine Commission (TLC), capturing trip records for yellow taxis, green taxis, and For-Hire Vehicles (FHV). The datasets encompass various fields such as pick-up and drop-off dates/times, locations, trip distances, fares, rate types, payment methods, and more.

## Data Sources

### Yellow Taxi Trips
Yellow taxis have been serving New York City since 2009 and are hailed via street signals or e-hail apps like Curb or Arro. They are permitted to respond to street hails across all five boroughs.

- **Data Fields**: pick-up and drop-off dates/times, locations, trip distances, itemized fares, rate types, payment types, and passenger counts.
- **Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)

### Green Taxi Trips
Introduced in August 2013, green taxis serve the boroughs of NYC and specific areas in Manhattan.

- **Data Fields**: pick-up and drop-off dates/times, locations, trip distances, itemized fares, rate types, payment types, and passenger counts.
- **Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf)

### For-Hire Vehicle (FHV) Data
FHV data includes trip records from high-volume for-hire vehicle bases, community livery bases, luxury limousine bases, and black car bases.

- **Data Fields**: dispatching base number, pick-up/drop-off dates/times, locations, trip distances, fare details, and more.
- **Data Dictionary**: [PDF Download](#)
- **FHV Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_fhv.pdf)
- **High Volume Data Dictionary**: [PDF Download](#https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf)

#### Note on Shared Rides
Shared rides information, like Lyft Line and Uber Pool, is included if the trip was specially reserved with one of these services.

#### Matching to FHV Bases
To identify the base that dispatched the trip, join the `dispatching_base_num` with the `License Number`. Note that high-volume bases might not match the commonly recognized company name.

- **Key for HVFHS Companies**: Juno, Lyft, Uber, and Via

**Disclaimer**: The TLC does not create the trip data and makes no representations regarding its accuracy.
