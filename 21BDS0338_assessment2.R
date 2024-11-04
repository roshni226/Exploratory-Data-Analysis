#Roshni Joshi- 21BDS0338
# Load necessary libraries
if (!require(tm)) install.packages("tm", dependencies = TRUE)
if (!require(textclean)) install.packages("textclean", dependencies = TRUE)
if (!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if (!require(text2vec)) install.packages("text2vec", dependencies = TRUE)
if (!require(caTools)) install.packages("caTools", dependencies = TRUE)
if (!require(caret)) install.packages("caret", dependencies = TRUE)

library(dplyr)
library(tm)
library(textclean)
library(text2vec)
library(caTools)
library(caret)

# 1. Load the dataset
imdb_data <- read.csv("C:/Users/Roshni/Downloads/IMDB Dataset.csv/IMDB Dataset.csv", stringsAsFactors = FALSE)

# Sample 1,000 rows from the dataset
set.seed(123)  # Setting seed for reproducibility
small_imdb_data <- imdb_data[sample(nrow(imdb_data), 400), ]

# Check the smaller dataset
head(small_imdb_data)

# Write the smaller dataset to a CSV file
write.csv(small_imdb_data, "small_IMDB_Dataset.csv", row.names = FALSE)

# 2. Data Preprocessing
# Convert text to lowercase
small_imdb_data$review <- tolower(small_imdb_data$review)

# Remove punctuation, numbers, and stop words
small_imdb_data$review <- removePunctuation(small_imdb_data$review)
small_imdb_data$review <- removeNumbers(small_imdb_data$review)

# Remove stop words
stopwords <- stopwords("en")
small_imdb_data$review <- removeWords(small_imdb_data$review, stopwords)

# Tokenize text
tokens <- small_imdb_data$review %>% 
  word_tokenizer()

# Create a vocabulary and vectorize text using text2vec
it <- itoken(tokens)
vectorizer <- vocab_vectorizer(vocabulary = create_vocabulary(it))
dtm <- create_dtm(it, vectorizer)

# 3. Convert sentiment to binary: positive = 1, negative = 0
small_imdb_data$sentiment <- ifelse(small_imdb_data$sentiment == "positive", 1, 0)

# 4. Split the data into training and testing sets
set.seed(123)
split <- sample.split(small_imdb_data$sentiment, SplitRatio = 0.8)
train_data <- small_imdb_data[split == TRUE, ]
test_data <- small_imdb_data[split == FALSE, ]

# Create training and testing DTM
train_dtm <- dtm[split == TRUE, ]
test_dtm <- dtm[split == FALSE, ]

# Convert the sparse matrices to data frames
train_dtm_df <- as.data.frame(as.matrix(train_dtm))
test_dtm_df <- as.data.frame(as.matrix(test_dtm))

# 5. Train the logistic regression model
# Include sentiment labels in the training data frame
train_dtm_df$sentiment <- train_data$sentiment

# Fit the logistic regression model
model <- glm(sentiment ~ ., data = train_dtm_df, family = "binomial")

# 6. Predict probabilities on the test set
predictions <- predict(model, newdata = test_dtm_df, type = "response")

# 7. Convert probabilities to binary classes
predicted_sentiment <- ifelse(predictions > 0.5, 1, 0)

# 8. Calculate accuracy
confusionMatrix(as.factor(predicted_sentiment), as.factor(test_data$sentiment))

