import findspark
findspark.init()
from pyspark import SparkContext,SparkConf
import collections

conf=SparkConf().setMaster("local").setAppName("RatingHistogram")

sc=SparkContext(conf=conf)

