from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, floor, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("CourseRecommendationSystem") \
    .config("spark.master", "local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

data_path = "complete_course_data.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

print("原始数据结构：")
df.printSchema()
print("原始数据示例：")
df.show(5)

df = df.withColumn("rating", col("rating").cast("float"))
num_courses = df.select("index").distinct().count()
num_users = num_courses * 5

df = df.withColumn("userId", (floor(rand() * num_users)).cast("int"))

ratings_df = df.select("userId", "index", "rating").withColumnRenamed("index", "courseId")

print("处理后的数据示例：")
ratings_df.show(5)

(training_df, test_df) = ratings_df.randomSplit([0.8, 0.2], seed=42)

print(f"训练集大小: {training_df.count()}")
print(f"测试集大小: {test_df.count()}")

als = ALS(
    userCol="userId",
    itemCol="courseId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    rank=10,
    maxIter=10,
    regParam=0.1
)

model = als.fit(training_df)
print("ALS 模型训练完成！")

predictions = model.transform(test_df)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

user_recs = model.recommendForAllUsers(5)
print("每个用户的推荐结果：")
user_recs.show(truncate=False)

course_recs = model.recommendForAllItems(5)
print("每门课程的推荐结果：")
course_recs.show(truncate=False)

model_save_path = "save_model"
model.save(model_save_path)
print(f"模型已保存到 {model_save_path}")

spark.stop()