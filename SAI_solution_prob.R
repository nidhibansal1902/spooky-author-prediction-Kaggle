# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library(xgboost)
library(Matrix)
library(data.table)
library(dplyr)


#read train and test csv
train<-read.csv("D:/Data Science/Kaggle/Spooky Author Identification/train.csv")
test<-read.csv("D:/Data Science/Kaggle/Spooky Author Identification/test.csv")
#sample_submission<-read.csv("D:/Data Science/Kaggle/Spooky Author Identification/sample_submission.csv")

#adding author column to test data frame
test$author<-NA

str(train)
head(train)

str(test)

train<-train[2:3]
test_data<-test[2:3]

#checking missing values
colSums(is.na(train))
colSums(train==" ")
colSums(is.na(test))
colSums(test==" ")


#function to create word cloud and dtm
text_func<-function(x){
  #create corpus and documet term matrix
  format_text<-Corpus(VectorSource(x))
  #inspect(format_text)
  
  #data processing
  format_text<-tm_map(format_text,content_transformer(tolower))
  format_text<-tm_map(format_text,removeNumbers)
  format_text<-tm_map(format_text,removePunctuation)
  format_text<-tm_map(format_text,removeWords,stopwords("English"))
  format_text<-tm_map(format_text,stripWhitespace)
  
  
  
  #memory.limit(size=100000)
  #Document term matrix
  dtm_full<-DocumentTermMatrix(format_text)
  dtm_full<-removeSparseTerms(dtm_full,0.99)
  
  dtm_matrix<-as.matrix(dtm_full)
  
  v_full <- sort(colSums(dtm_matrix),decreasing=TRUE)
  
  
  
  words_full <- names(v_full)
  d_full <- data.frame(word=words_full, freq=v_full)
  pal <-brewer.pal(8,"Dark2")
  wordcloud(d_full$word,d_full$freq,color = pal,random.order = FALSE,rot.per = 0.3)
  
  return(as.data.frame(dtm_matrix))
}

#creating word cloud
d_train<-text_func(train$text)
d_test<-text_func(test$text)
d_HPL<-text_func(train$text[train$author=="HPL"])
d_MWS<-text_func(train$text[train$author=="MWS"])
d_EAP<-text_func(train$text[train$author=="EAP"])

#xgboost modeling

d_train$author <- as.factor(c(train$author))
d_test$author<-as.factor(rep('EAP',nrow(d_test)))

#Preparing dataset
ctrain <- xgb.DMatrix(Matrix(data.matrix(d_train[,!colnames(d_train) %in% c('author')])), label = as.numeric(d_train$author)-1)

dtest <- xgb.DMatrix(Matrix(data.matrix(d_test[,!colnames(d_test) %in% c('author')])) )



#train multiclass model using softmax
#params
xgb_params = list(
  objective = "multi:softprob",
  num_class = 3,
  booster = "gbtree",
  eta = 0.1,
  max_depth = 6,
  subsample = 1
)


#model
xgb_model <- xgb.train(data = ctrain,
                       params = xgb_params,
                       nrounds = 100,
                       verbose = 1)

#predict
xgb_model.predict <- predict(xgb_model, newdata = dtest) %>% matrix(ncol = 3)

final_submit <- cbind(test["id"], xgb_model.predict)

colnames(final_submit) <- c("id", "EAP", "HPL", "MWS")


#final submission
write.csv(final_submit, 'D:/Data Science/Kaggle/Spooky Author Identification/Submission.csv', row.names = FALSE)