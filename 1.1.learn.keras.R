##### https://www.datacamp.com/community/tutorials/keras-r-deep-learning #####

install.packages("keras")
library(keras)
install_keras() 

install.packages("corrplot")
library(corrplot)

### read example CSV file ###
f = url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
iris <- read.csv(f, header = F, as.is = T) 
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
iris$Species = as.factor(iris$Species)

### view data ###
plot(iris$Petal.Length, 
     iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
     xlab="Petal Length", 
     ylab="Petal Width")

corrplot(cor(iris[,1:4]),'square')

### normalization or not ###
menorm <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

# Normalize the `iris` data, my norm
iris_norm <- as.data.frame(lapply(iris[1:4], menorm))

# Normalize the `iris` data, keras
iris[,5] <- as.numeric(iris[,5]) -1
iris <- as.matrix(iris)
dimnames(iris) <- NULL

iris_korm <- normalize(iris[,1:4])
dim(iris_norm); dim(iris_korm)

# compare normalization
head(iris_norm)
head(iris_korm)
cor(iris[,1],iris_korm[,1])

### divide into training and validation sets ###
set.seed(1433)
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]

# prepare labels using ONE HOT Encode
iris.trainingtarget <- iris[ind==1, 5]
iris.testtarget <- iris[ind==2, 5]
iris.trainLabels <- to_categorical(iris.trainingtarget)
iris.testLabels <- to_categorical(iris.testtarget)
print(iris.testLabels)

### run MLP model ###
# make a sequential model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 7, activation = 'relu') %>%
  layer_dense(units = 4, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# OR, define an optimizer
#sgd <- optimizer_sgd(lr = 0.01)
#model %>% compile(optimizer=sgd, 
#                  loss='categorical_crossentropy', 
#                  metrics='accuracy')

# OR, more optimizer
#opt = optimizer_rmsprop(lr = 0.001, rho = 0.9, epsilon = NULL, decay = 0.0)
#model %>% compile(optimizer=opt, 
#                  loss='categorical_crossentropy', 
#                  metrics='accuracy')

# Fit the model 
model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 100, 
  batch_size = 5, 
  validation_split = 0.2
)

# Predict the classes for the test data
model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
classes <- model %>% predict_classes(iris.test, batch_size = 128)
# Confusion matrix
table(iris.testtarget, classes)

