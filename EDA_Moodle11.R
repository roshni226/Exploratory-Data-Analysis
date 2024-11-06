# ROSHNI JOSHI - 21BDS0338
# Load necessary libraries
library(class)
library(dplyr)
library(ggplot2)
library(caret)  # For confusion matrix visualization

# Use the built-in iris dataset
data <- iris

# Display the first few rows of the dataset
head(data)

# Normalize function to scale the features between 0 and 1
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization only to feature columns
data_normalized <- as.data.frame(lapply(data[, 1:4], normalize))

# Add target variable back to the normalized data frame
data_normalized$Species <- data$Species

# Improved user input handling with default values
cat("Enter the value of k (e.g., 3): ")
input_k <- readline()
k <- ifelse(nzchar(input_k), as.integer(input_k), 3)  # Default to 3 if no input

cat("Enter the training data ratio (e.g., 0.8 for 80%): ")
input_ratio <- readline()
train_ratio <- ifelse(nzchar(input_ratio), as.numeric(input_ratio), 0.8)  # Default to 0.8 if no input

# Ensure valid inputs
if (is.na(k) || k <= 0) {
  k <- 3
  cat("Invalid k value, defaulting to k = 3\n")
}
if (is.na(train_ratio) || train_ratio <= 0 || train_ratio >= 1) {
  train_ratio <- 0.8
  cat("Invalid training ratio, defaulting to 0.8\n")
}

# Split dataset into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(data_normalized), train_ratio * nrow(data_normalized))
train_data <- data_normalized[train_indices, ]
test_data <- data_normalized[-train_indices, ]

# Separate features and target variables
train_features <- train_data[, -ncol(train_data)]
train_labels <- train_data$Species
test_features <- test_data[, -ncol(test_data)]
test_labels <- test_data$Species

# Apply KNN model
knn_pred <- knn(train = train_features, test = test_features, cl = train_labels, k = k)

# Model evaluation: Calculate accuracy
accuracy <- sum(knn_pred == test_labels) / length(test_labels) * 100
cat("Accuracy of the KNN model with k =", k, "is", round(accuracy, 2), "%\n")

# Create a confusion matrix
conf_matrix <- table(Predicted = knn_pred, Actual = test_labels)
cat("Confusion Matrix:\n")
print(conf_matrix)

# Visualize the confusion matrix using ggplot2 as a heatmap
conf_matrix_df <- as.data.frame(conf_matrix)
ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "red", high = "purple") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = paste("Confusion Matrix (k =", k, ")"),
       x = "Actual Species",
       y = "Predicted Species") +
  theme_minimal()

# Plot the predicted vs actual values
ggplot(data.frame(Test_Label = test_labels, Predicted_Label = knn_pred), aes(x = Test_Label, fill = Predicted_Label)) +
  geom_bar(position = "dodge") +
  labs(title = paste("KNN Prediction Results with k =", k),
       x = "Actual Species",
       y = "Count") +
  scale_fill_manual(values = c("setosa" = "pink", "versicolor" = "lightgreen", "virginica" = "yellow")) +
  theme_minimal()

