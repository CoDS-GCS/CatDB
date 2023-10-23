package ca.concordia;

import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public class StatisticalQueryByDF {

    public static void main(String[] args) {

        String datasetRoot = args[0];
        String query = args[1];

        DataSource ds = new DataSource(datasetRoot);

        SparkSession spark = SparkSession.builder().appName("csv-by-sql-demo").getOrCreate();
        Map<String, String> yelpTBLName = ds.getYELPDatasets();

        Dataset<Row> review = spark.read().json(yelpTBLName.get("review"));

//        if (query.equals("Q1")){
//            df = doJoin(spark, business, review, user);
//        }
//        else if (query.equals("Q2")){
//            df = doJoin(spark, business, review, user);
//            df = df.filter("r_stars<=5");
//        }
//        else if (query.equals("Q3")){
//            df = doJoin(spark, business, review, user);
//            df = df.filter("r_stars=2");
//        }
//        else if (query.equals("Q4")){
//            df = doJoin(spark, business, review, user);
//            df = df.filter("r_stars =2 or r_stars=5");
//        }
//        else if (query.equals("Q5")){
//            review = review.filter("r_stars =2");
//            df = doJoin(spark, business, review, user);
//        }
//        AtomicLong count = new AtomicLong();
//        df.foreach((ForeachFunction<Row>) row -> count.getAndIncrement());
    }

    public static Dataset<Row> doJoin(SparkSession spark,Dataset<Row> business, Dataset<Row> review, Dataset<Row> user){
        Dataset<Row> df = mergeDataFrames(business, review, "business_id");
        df = mergeDataFrames(df, user, "user_id");
        return df;
    }

    public static Dataset<Row> mergeDataFrames(Dataset<Row> df1, Dataset<Row> df2, String col_name) {
        return df1.join(df2, col_name, "left");
    }
}
