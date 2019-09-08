library(sager)
library(reticulate)
library(stringr)
library(readr)
library(ggplot2)
library(dplyr)
library(configr)

# Usefule functions


# virtualenv_create('sage')

# virtualenv_install(
#   'sage',
#   c(
#     'sagemaker',
#     'boto3',
#     'numpy',
#     'pandas'
#   ),
#   ignore_installed = TRUE
# )

config_data = configr::read.config('config.ini')

use_python('/Users/digitalfirstmedia/.virtualenvs/sage/bin/python')

configure_aws(
  config_data$aws$user,
  config_data$aws$pass,
  config_data$aws$region
)




# AKIAWEUHS5MES75VBIOD
# t/cCQ2o/sIkiHx8t7Tt8K2IL7T6vOKqlGznHPlFJ

sagemaker <- import('sagemaker')
session <- sagemaker$Session()
bucket <- session$default_bucket()
role_arn <- sagemaker$get_execution_role()







data_file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
abalone <- read_csv(file = data_file, col_names = FALSE)
names(abalone) <- c('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
head(abalone)



abalone$sex <- as.factor(abalone$sex)
summary(abalone)



ggplot(abalone, aes(x = height, y = rings, color = sex)) + geom_point() + geom_jitter()

abalone <- abalone %>%
  filter(height != 0)


abalone <- abalone %>%
  mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
         male = as.integer(ifelse(sex == 'M', 1, 0)),
         infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
  select(-sex)
abalone <- abalone %>%
  select(rings:infant, length:shell_weight)
head(abalone)





abalone_train <- abalone %>%
  sample_frac(size = 0.7)
abalone <- anti_join(abalone, abalone_train)
abalone_test <- abalone %>%
  sample_frac(size = 0.5)
abalone_valid <- anti_join(abalone, abalone_test)




write_csv(abalone_train, 'abalone_train.csv', col_names = FALSE)
write_csv(abalone_valid, 'abalone_valid.csv', col_names = FALSE)


s3_train <- session$upload_data(path = 'abalone_train.csv',
                                bucket = bucket,
                                key_prefix = 'data')
s3_valid <- session$upload_data(path = 'abalone_valid.csv',
                                bucket = bucket,
                                key_prefix = 'data')


abalone_valid <- anti_join(abalone, abalone_test)




s3_train_input <- sagemaker$s3_input(s3_data = s3_train,
                                     content_type = 'csv')
s3_valid_input <- sagemaker$s3_input(s3_data = s3_valid,
                                     content_type = 'csv')

registry <- sagemaker$amazon$amazon_estimator$registry(session$boto_region_name, algorithm='xgboost')
container <- paste(registry, '/xgboost:latest', sep='')

s3_output <- paste0('s3://', bucket, '/output')
estimator <- sagemaker$estimator$Estimator(image_name = container,
                                           role = role_arn,
                                           train_instance_count = 1L,
                                           train_instance_type = 'ml.m5.4xlarge',
                                           train_volume_size = 30L,
                                           train_max_run = 3600L,
                                           input_mode = 'File',
                                           output_path = s3_output,
                                           output_kms_key = NULL,
                                           base_job_name = NULL,
                                           sagemaker_session = NULL)



estimator$set_hyperparameters(num_round = 40L)
job_name <- paste('sagemaker-train-xgboost', format(Sys.time(), '%H-%M-%S'), sep = '-')
input_data <- list('train' = s3_train_input,
                   'validation' = s3_valid_input)
estimator$fit(inputs = input_data,
              job_name = job_name)

estimator$model_data


model_endpoint <- estimator$deploy(initial_instance_count = 1L,
                                   instance_type = 'ml.t2.medium')

model_endpoint$content_type <- 'text/csv'
model_endpoint$serializer <- sagemaker$predictor$csv_serializer


abalone_test <- abalone_test[-1]
num_predict_rows <- 500
test_sample <- as.matrix(abalone_test[1:num_predict_rows, ])
dimnames(test_sample)[[2]] <- NULL



predictions <- model_endpoint$predict(test_sample)
predictions <- str_split(predictions, pattern = ',', simplify = TRUE)
predictions <- as.numeric(predictions)


abalone_test <- cbind(predicted_rings = predictions,
                      abalone_test[1:num_predict_rows, ])
head(abalone_test)


session$delete_endpoint(model_endpoint$endpoint)
