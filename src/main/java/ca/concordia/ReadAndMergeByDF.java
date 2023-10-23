package ca.concordia;

import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public class ReadAndMergeByDF {

    public static void main(String[] args) {

        String datasetRoot = args[0];
        String query = args[1];

        DataSource ds = new DataSource(datasetRoot);

        SparkSession spark = SparkSession.builder().appName("csv-by-sql-demo").getOrCreate();
        Map<String, String> yelpTBLName = ds.getYELPDatasets();

        Dataset<Row> business = spark.read().json(yelpTBLName.get("business"))
                .withColumnRenamed("stars", "b_stars")
                .withColumnRenamed("name","b_name");
        Dataset<Row> review = spark.read().json(yelpTBLName.get("review"))
                .withColumnRenamed("stars", "r_stars");
        Dataset<Row> user = spark.read().json(yelpTBLName.get("user"))
                .withColumnRenamed("name", "u_name");;

        Dataset<Row> df = null;
        if (query.equals("Q1")){
            df = mergeDataFrames(review, user, "user_id");
            df = mergeDataFrames(df,business,"business_id");
        }
        else if (query.equals("Q2")){
            df = mergeDataFrames(review, user, "user_id");
            df = mergeDataFrames(df,business,"business_id");
            df = df.filter("r_stars<=5");
        }
        else if (query.equals("Q3")){
            df = mergeDataFrames(review, user, "user_id");
            df = mergeDataFrames(df,business,"business_id");
            df = df.filter("r_stars=2");
        }
        else if (query.equals("Q4")){
            df = mergeDataFrames(review, user, "user_id");
            df = mergeDataFrames(df,business,"business_id");
            df = df.filter("r_stars =2 or r_stars=5");
        }
        else if (query.equals("Q5")){
            review = review.filter("r_stars =2");
            df = mergeDataFrames(review, user, "user_id");
            df = mergeDataFrames(df,business,"business_id");
        }
        AtomicLong count = new AtomicLong();
        df.foreach((ForeachFunction<Row>) row -> count.getAndIncrement());
    }

    public static Dataset<Row> mergeDataFrames(Dataset<Row> df1, Dataset<Row> df2, String col_name) {
        return df1.join(df2, col_name, "left");
    }
}
