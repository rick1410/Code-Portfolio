library(dplyr)
library(RPostgres)
library(highfrequency)
library(lubridate)
library(data.table)
library(xts)
library(ggplot2)
library(zoo)

# Connect to WRDS database
wrds <- dbConnect(Postgres(),
                  host = 'wrds-pgdata.wharton.upenn.edu',
                  port = 9737,
                  dbname = 'wrds',
                  sslmode = 'require',
                  user = 'rickteeuwissen')

# Query the available dates
res <- dbSendQuery(wrds, "select distinct table_name
                   from information_schema.columns
                   where table_schema='taqmsec'
                   order by table_name")
df_dates <- dbFetch(res, n = -1)
dbClearResult(res)

# Store the available dates
dates_trades <- df_dates %>%
  filter(grepl("ctm", table_name), !grepl("ix_ctm", table_name)) %>%
  mutate(table_name = substr(table_name, 5, 12)) %>%
  filter(nchar(table_name) == 8) %>%  
  unlist()

# Select trading dates within the desired range
start_date <- "20171231"
end_date <- "20250301"
trading_days <- dates_trades[dates_trades >= start_date & dates_trades <= end_date]
# Generate regularized time series
trading_days_dates <- as.Date(trading_days, format = "%Y%m%d")



# Define stocks
stocks <- c("AMD", "NVDA", "INTC", "MSFT", "AAPL", "ADBE", 
            "TSLA", "IBM", "SMCI", "TSM", "ASML", "ANET", "AMZN")

# Define data folder
data_folder <- "C:/Users/rickt/Desktop/Econometrics and Data Science/Thesis/Data/Stock/High Frequency"



# Process each stock
for (stock in stocks) {
  stock_dir <- file.path(data_folder, stock)
  cleaned_filename <- file.path(stock_dir, paste0("cleaned_data_", stock, ".csv"))
  
  if (!file.exists(cleaned_filename)) {
    print(paste("Skipping:", stock, "- cleaned data not found"))
    next
  }
  
  # Read cleaned stock data
  cleaned_data <- fread(cleaned_filename)
  
  
  adjusted_dates <- unique(as.Date(cleaned_data$DT))
  filtered_trading_days <- trading_days_dates[trading_days_dates %in% adjusted_dates]
  
  trading_seconds_dt <- rbindlist(
    lapply(filtered_trading_days, function(day) {
      start_time <- as.POSIXct(paste(day, "09:30:00"), tz = "UTC")
      end_time <- as.POSIXct(paste(day, "16:00:00"), tz = "UTC")
      seconds <- seq(from = start_time, to = end_time, by = "1 sec")
      data.table(DT = seconds)
    })
  )
  cleaned_data[, DT := as.POSIXct(DT, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")]
  trading_seconds_dt[, DT := as.POSIXct(DT, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")]
  
  filled_data <- trading_seconds_dt[, .(DT)] %>%
    merge(cleaned_data, by = "DT", all.x = TRUE)
  
  filled_data[, PRICE := zoo::na.locf(PRICE, na.rm = FALSE)]  # Forward fill
  filled_data[, PRICE := zoo::na.locf(PRICE, fromLast = TRUE)]  # Backward fill
  
  # Save filled (regular) data to CSV
  filled_filename <- file.path(stock_dir, paste0("filled_data_", stock, ".csv"))
  fwrite(filled_data, filled_filename)
  print(paste("Saved:", filled_filename))
}

print("All stocks processed successfully!")
