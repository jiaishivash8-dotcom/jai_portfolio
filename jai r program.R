# ------------------------------
# 1. Install and load libraries
# ------------------------------
install.packages("caret")
install.packages("e1071")     # Required by caret
install.packages("dplyr")

library(caret)
library(e1071)
library(dplyr)


# ----------------------------------------
# 2. Load dataset (correct file path)
# ----------------------------------------
# Use double backslashes or forward slashes in file path
auto_data <- read.csv("C:/Users/DELL/Downloads/indianautomobiles.csv",
                      header = TRUE, stringsAsFactors = TRUE)

# ----------------------------------------
# 3. View and clean the dataset
# ----------------------------------------
# Display structure and first few rows
str(auto_data)
head(auto_data)

# Remove rows with missing values
auto_data <- na.omit(auto_data)

# ----------------------------------------
# 4. Define target variable
# ----------------------------------------
# Replace 'Price' with your actual target column name
# For example: auto_data$Price
# Check column names: 
names(auto_data)

# Assuming we are predicting 'Price'
# If 'Price' is not the target, change accordingly
target <- "Price"

# ----------------------------------------
# 5. Split data into training and testing sets
# ----------------------------------------
set.seed(123)  # For reproducibility

trainIndex <- createDataPartition(auto_data[[target]], p = 0.7, list = FALSE)
train_data <- auto_data[trainIndex, ]
test_data <- auto_data[-trainIndex, ]

# ----------------------------------------
# 6. Train a linear regression model
# ----------------------------------------
model <- train(as.formula(paste(target, "~ .")),
               data = train_data,
               method = "lm")

# Print model summary
print(model)

# ----------------------------------------
# 7. Make predictions on test set
# ----------------------------------------
predictions <- predict(model, newdata = test_data)

# ----------------------------------------
# 8. Evaluate the model
# ----------------------------------------
# Calculate RMSE and R-squared
results <- postResample(predictions, test_data[[target]])

print(results
