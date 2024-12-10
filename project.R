if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(dslabs)) install.packages("dslabs")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")
if(!require(broom)) install.packages("broom")
if(!require(purrr)) install.packages("purrr")

#Download the CSV file to a temporary file
url <- "https://github.com/KelvinMg/edx_ml_project/raw/refs/heads/main/student_monnitoring_data.csv"  
temp_file <- tempfile(fileext = ".csv")
?download.file
download.file(url, temp_file, mode = "w")

#Read the contents of the CSV into an R object
data <- read.csv(temp_file) 

#Delete the temporary file
file.remove(temp_file)

#Verify the data
head(data)
View(data)
str(data)
sum(is.na(data))


library(tidyverse)
library(dslabs)
library(ggplot2)
library(dplyr)
library(caret)
library(broom)
library(purrr)


data
str(data)

#making factor variables 
data <- data %>% mutate(Attendance.Status = factor(Attendance.Status),
                        Risk.Level = factor(Risk.Level))

#summary of each student id
result <- data %>%
  group_by(Student.ID, Attendance.Status) %>%
  summarise(
    Mean_Stress_Level = mean(Stress.Level..GSR., na.rm = TRUE),
    Mean_Sleep_Hours = mean(Sleep.Hours, na.rm = TRUE),
    Mean_Anxiety_Level = mean(Anxiety.Level, na.rm = TRUE),
    Mean_Mood_Score = mean(Mood.Score, na.rm = TRUE)
  )

#creating a data frame for more analysis and prediction
data_summary <- as.data.frame(result) 
str(data_summary)

#we are going to ignore late as we are looking to predict whether
#a student will attend or not 
data_summary <- data_summary %>% filter(Attendance.Status != "Late") %>%
  mutate(Attendance.Status = factor(Attendance.Status))

str(data_summary)

sum(data_summary$Attendance.Status == "Present")
#number of present cases is 500

sum(data_summary$Attendance.Status == "Absent")
#number of absent cases is 500
#the data is well balanced

#I will try and predict attendance status based on the features below
#exploratory analysis

#first feature is stress level
#seems to be normally distributed
data_summary %>% ggplot(aes(Mean_Stress_Level)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.5, color = "black", fill = "skyblue") +
  geom_density(color = "red", linewidth = 1) +
  theme_minimal() +
  labs(title = "Histogram with Density Curve", x = "Value", y = "Density")

#Boxplot for stress level by attendance status
ggplot(data_summary, aes(x = Attendance.Status, y = Mean_Stress_Level)) +
  geom_boxplot() 
#when present students have slightly higher stress than when absent
#which could indicate when students are present they have accumulated stress due to their regular school attendance and 
#responsibilities while when they are absent they might not be participating in these stressful situations, and as a result, 
#their stress levels may not reflect the same intensity.


#sleep hours
#data is normally distributed
data_summary %>% ggplot(aes(Mean_Sleep_Hours)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.5, color = "black", fill = "skyblue") +
  geom_density(color = "red", linewidth = 1) +
  theme_minimal() +
  labs(title = "Histogram with Density Curve", x = "Value", y = "Density")

#Boxplot for Sleep hours by Attendance Status
ggplot(data_summary, aes(x = Attendance.Status, y = Mean_Sleep_Hours)) +
  geom_boxplot() 
#when students are present they seem to have had slightly better sleep hours when absent 


#anxiety level
#the data is normally distributed
data_summary %>% ggplot(aes(Mean_Anxiety_Level)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.5, color = "black", fill = "skyblue") +
  geom_density(color = "red", linewidth = 1) +
  theme_minimal() +
  labs(title = "Histogram with Density Curve", x = "Value", y = "Density")

#boxplot for anxiety levels by attendance status
data_summary %>% ggplot(aes(x  = Attendance.Status,y = Mean_Anxiety_Level)) +
  geom_boxplot()
#when students are present they have lower anxiety levels than when absent suggesting that when students are absent they
#might be avoiding the situations that trigger their anxiety. Their absence might be reflecting their anxiety about school,
#and being absent might provide temporary relief from this anxiety. On the other hand when students are present they may be 
#actively confronting and managing their anxiety through exposure,social interaction, and academic engagement, which can help 
#reduce anxiety.

#mood score
#the data is normally distributed
data_summary %>% ggplot(aes(Mean_Mood_Score)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.5, color = "black", fill = "skyblue") +
  geom_density(color = "red", linewidth = 1) +
  theme_minimal() +
  labs(title = "Histogram with Density Curve", x = "Value", y = "Density")

data_summary %>% ggplot(aes(x = Attendance.Status,y = Mean_Mood_Score)) +
  geom_boxplot()
#The higher mood in students when they are present compared to when they are absent suggests that attendance at school is
#associated with a more positive emotional state, likely due to social engagement, routine, academic participation, and 
#the opportunity for positive emotional experiences at school. In contrast, when students are absent they may be experiencing
#lower mood due to isolation, avoidance, or struggles with emotional or psychological challenges, which could lead them to miss
#out on the positive mood-regulating benefits of school life.



#correlation between the features
cor_matrix <- cor(data_summary %>% select(Mean_Stress_Level, Mean_Sleep_Hours, Mean_Anxiety_Level, Mean_Mood_Score))
cor_matrix

#results of the correlation on the dataset
#the amount of sleep a student gets doesn’t seem to have a noticeable impact on their reported stress level.
#students' stress levels don’t seem to be closely related to their anxiety levels on average.
#students who experience higher stress also report higher mood scores, but the relationship is not strong.
#The amount of sleep a student gets doesn’t significantly impact their anxiety level on average.
#the number of hours a student sleeps doesn’t noticeably influence their reported mood.

#The correlations between these features are generally weak, which means that stress levels, sleep hours, 
#anxiety levels, and mood scores do not show strong relationships with one another. They might be influenced 
#by other factors not captured in this dataset, or there may be other complexities (e.g., non-linear relationships) 
#that aren't captured by simple correlation.

#Mood appears to slightly increase with stress (positive correlation of 0.0412).
#Anxiety and sleep hours show a very weak negative relationship with mood and sleep, respectively,
#but these trends are not strong enough to be of high significance.

#correlation with attendance
cor(data_summary %>% mutate(Attendance.Status = ifelse(Attendance.Status == "Absent", 0, 1)) %>%
                            select(-Student.ID))

#There is a slight tendency for attendance to be positively correlated with mood and stress, but these correlations are minimal.



#machine learning
y <- data_summary$Attendance.Status

#generate training and test sets
set.seed(42)
test_index <- createDataPartition(y, times = 1, p = 0.3, list = FALSE)
test_set <- data_summary[test_index, ]
train_set <- data_summary[-test_index, ]

y <- test_set$Attendance.Status


#to compare with later models we will check the accuracy if you randomly guess
y_hat <- sample(c("Present", "Absent"), length(test_index), replace = TRUE) %>% 
  factor(levels = levels(data_summary$Attendance.Status))

guessing_accuracy <- mean(y_hat == y)
guessing_accuracy 
guessing_conf_matrix <- confusionMatrix(y_hat, y)
guessing_conf_matrix

#guessing RMSE
y_hat_guessing_rmse <- as.integer( ifelse(y_hat == "Present", 1, 0))
y_rmse <- as.integer(ifelse(y == "Present", 1,0))
RMSE_guessing_model <- sqrt(mean((y_hat_guessing_rmse - y_rmse)^2))
RMSE_guessing_model

#rmse is far from zero meaning the predicted values are not close to the true values
#which is expected since we are guessing 

#algorithm to understand which feature predicts attendance status better

str(train_set)
y <- test_set$Attendance.Status
accuracy <- function(x){
  rangedValues <- seq(range(x)[1], range(x)[2], by=0.1)
  sapply(rangedValues, function(i){
    y_hat <- ifelse(x>i, 'Present', 'Absent')
    mean(y_hat==y)
  })
}

predictions <- apply(train_set[,-c(1,2)], 2, accuracy)
sapply(predictions, max)




guessing_conf_matrix$overall["Accuracy"]
#seems there is no 1 feature that can predict better than guessing as they
#fall within the confidence interval of the guessing model accuracy between 0.46 and 0.57
#best feature out of all the features is moodscore


#first model will be logistic regression
#the features have extremely weak correlations with the attendance status but the model might still find some patterns in the data.

log_model <- glm(Attendance.Status ~ Mean_Mood_Score + Mean_Anxiety_Level + Mean_Stress_Level + Mean_Sleep_Hours, 
                 family = binomial, 
                 data = train_set)


p_hat_glm <- predict(log_model, newdata = test_set, type = "response")
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, 1, 0), levels = c(0, 1), labels = c("Absent", "Present"))

log_model_conf_matrix <- confusionMatrix(y_hat_glm, y)

log_model_conf_matrix
guessing_conf_matrix

#rmse
y_hat_logmodel_rmse <- as.integer(ifelse(y_hat_glm == "Present", 1, 0))
RMSE_logmodel <- sqrt(mean((y_hat_logmodel_rmse - y_rmse)^2))
RMSE_logmodel
RMSE_guessing_model

#The model is not better than guessing, It has worse accuracy and worse sensitivity
#since we are trying to predict absent

summary(log_model)
#the summary shows only moodscore is statistically significant since it's p value is <0.05
#so we will use moodscore separately after attempting to predict with all the other predictors 
#in all the models we test onwards.
#using only mean_mood_score
log_model_mood <- glm(Attendance.Status ~ Mean_Mood_Score, 
                 family = binomial, 
                 data = train_set)

p_hat_glm_mood <- predict(log_model_mood, newdata = test_set, type = "response")
y_hat_glm_mood <- factor(ifelse(p_hat_glm_mood > 0.55, 1, 0), levels = c(0, 1), labels = c("Absent", "Present"))
log_model_mood_conf_matrix <- confusionMatrix(y_hat_glm_mood, y)
log_model_mood_conf_matrix

#rmse
y_hat_logmodelmood_rmse <- as.integer(ifelse(y_hat_glm_mood == "Present", 1, 0))
RMSE_logmodelmood <- sqrt(mean((y_hat_logmodelmood_rmse - y_rmse)^2))
RMSE_logmodelmood
RMSE_guessing_model

#The only advantage the model has over guessing is sensitivity

#carret package train
train_glm <- train(Attendance.Status ~ Mean_Mood_Score + Mean_Anxiety_Level + Mean_Stress_Level + Mean_Sleep_Hours, method = "glm", data = train_set)
y_hat_train_glm <- predict(train_glm, test_set, type = "raw")
log_model_train_conf_matrix <- confusionMatrix(y_hat_train_glm,y)
log_model_train_conf_matrix

#rmse
y_hat_train_glm_rmse <- as.integer(ifelse(y_hat_train_glm=="Present", 1, 0))
RMSE_logmodeltraining <- sqrt(mean((y_hat_train_glm_rmse - y_rmse)^2))
RMSE_logmodeltraining
RMSE_guessing_model

#has no advantage on guessing
#using only mood
train_glm <- train(Attendance.Status ~ Mean_Mood_Score, method = "glm", data = train_set)
y_hat_trainmood_glm <- predict(train_glm, test_set, type = "raw")
log_model_mood_train_conf_matrix <- confusionMatrix(y_hat_trainmood_glm,y)
log_model_mood_train_conf_matrix

#rmse
y_hat_trainmood_glm_rmse <- as.integer(ifelse(y_hat_trainmood_glm=="Present", 1, 0))
RMSE_logmodelmoodtraining <- sqrt(mean((y_hat_trainmood_glm_rmse - y_rmse)^2))
RMSE_logmodelmoodtraining
RMSE_guessing_model

#has no advantage on guessing, all logistic regression models tested above had lower accuracy
#with only one using mood and not utilizing the train function had adantage in sensitivity
#but on ther hand an accuracy rate below 50%


#second model will be  k-Nearest Neighbors with cross validation
#choosing k
ks <- seq(3, 500, 2)

accuracy <- map_df(ks, function(k){
  fit <- knn3(Attendance.Status ~ Mean_Mood_Score + Mean_Anxiety_Level + Mean_Stress_Level + Mean_Sleep_Hours,
               data = train_set, k = k)
  # Predict on the training set
  y_hat_k_train <- predict(fit, train_set, type = "class")
  cm_train <- confusionMatrix(y_hat_k_train, train_set$Attendance.Status)
  train_error <- cm_train$overall["Accuracy"]
  
  # Predict on the test set
  y_hat_k_test <- predict(fit, test_set, type = "class")
  cm_test <- confusionMatrix(y_hat_k_test, y)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})

accuracy %>% mutate(k = ks) %>%
  gather(set, accuracy, -k) %>%
  mutate(set = factor(set, levels = c("train", "test"))) %>%
  ggplot(aes(k, accuracy, color = set)) + 
  geom_line() +
  geom_point()

best_k <- ks[which.max(accuracy$test)]
best_K_accuaracy <- max(accuracy$test)

best_k
best_K_accuaracy

#with only the mood score 
ks <- seq(3, 500, 2)

accuracy_mood <- map_df(ks, function(k){
  fit <- knn3(Attendance.Status ~ Mean_Mood_Score,
              data = train_set, k = k)
  
  # Predict on the training set
  y_hat_k_train <- predict(fit, train_set, type = "class")
  cm_train <- confusionMatrix(y_hat_k_train, train_set$Attendance.Status)
  train_error <- cm_train$overall["Accuracy"]
  
  # Predict on the test set
  y_hat_k_test <- predict(fit, test_set, type = "class")
  cm_test <- confusionMatrix(y_hat_k_test, y)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})

accuracy_mood %>% mutate(k = ks) %>%
  gather(set, accuracy, -k) %>%
  mutate(set = factor(set, levels = c("train", "test"))) %>%
  ggplot(aes(k, accuracy, color = set)) + 
  geom_line() +
  geom_point()

best_kmood <- ks[which.max(accuracy_mood$test)]
best_kmood_accuracy <- max(accuracy_mood$test)
best_kmood
best_kmood_accuracy

#evaluating model
#training
knn_fit <- knn3(Attendance.Status ~ Mean_Mood_Score + Mean_Anxiety_Level + Mean_Stress_Level + Mean_Sleep_Hours,
            data = train_set, k = best_k)


y_hat_knn <- predict(knn_fit, test_set, type = "class")

knn_conf_matrix <- confusionMatrix(y_hat_knn, y)

knn_conf_matrix
guessing_conf_matrix

#rmse
y_hat_knn_rmse <- as.integer(ifelse(y_hat_knn == "Present", 1, 0))
RMSE_knn <- sqrt(mean((y_hat_knn_rmse - y_rmse)^2))
RMSE_knn
RMSE_guessing_model

#the model is better than guessing in both sensitivity and accuracy but not a huge difference
#using only mood score
#training

knn_fitmood <- knn3(Attendance.Status ~ Mean_Mood_Score,
                data = train_set, k = best_kmood)


y_hat_knnmood <- predict(knn_fitmood, test_set, type = "class")

knnmood_conf_matrix <- confusionMatrix(y_hat_knnmood, y)

knnmood_conf_matrix
guessing_conf_matrix

#rmse
y_hat_knnmood_rmse <- as.integer(ifelse(y_hat_knnmood == "Present", 1, 0))
RMSE_knnmood <- sqrt(mean((y_hat_knn_rmse - y_rmse)^2))
RMSE_knnmood
RMSE_knn
RMSE_guessing_model

knn_conf_matrix
knnmood_conf_matrix
guessing_conf_matrix

#caret train
#train
control <- trainControl(method = "cv", number = 10, p = .9)
tune_grid <- expand.grid(k = best_k)
train_knn <- train(Attendance.Status ~ Mean_Mood_Score + Mean_Anxiety_Level + Mean_Stress_Level + Mean_Sleep_Hours,
                   method = "knn", data = train_set, tuneGrid = tune_grid, trControl = control)


y_hat_knntrain <- predict(train_knn, test_set, type = "raw")

knntrain_conf_matrix <- confusionMatrix(y_hat_knntrain, y)
knntrain_conf_matrix

#rmse
y_hat_knntrain_rmse <- as.integer(ifelse(y_hat_knntrain=="Present", 1, 0))
RMSE_knntrain <- sqrt(mean((y_hat_knntrain_rmse - y_rmse)^2))
RMSE_knntrain

#using only mood score
control1 <- trainControl(method = "cv", number = 10, p = .9)
tune_grid1 <- expand.grid(k = best_kmood)
train_knnmood <- train(Attendance.Status ~ Mean_Mood_Score,
                       method = "knn", data = train_set, tuneGrid = tune_grid1,
                       trControl = control1) 

y_hat_knnmoodtrain <- predict(train_knnmood, test_set, type = "raw")
knnmoodtrain_conf_matrix <- confusionMatrix(y_hat_knnmoodtrain, y)
knnmoodtrain_conf_matrix

#rmse
y_hat_knnmoodtrain_rmse <- as.integer(ifelse(y_hat_knnmoodtrain=="Present", 1, 0))
RMSE_knnmoodtrain <- sqrt(mean((y_hat_knnmoodtrain_rmse - y_rmse)^2))
RMSE_knnmoodtrain
RMSE_knntrain
RMSE_knnmood
RMSE_knntrain
RMSE_guessing_model
#the knn model that used only mood score (train_knnmood) is the best of all the models tested
#with the highest accuracy and sensitivity for Absent. It better than guessing but
#not strong models.





