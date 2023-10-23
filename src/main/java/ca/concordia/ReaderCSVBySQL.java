package ca.concordia;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;

public class ReaderCSVBySQL {

    public static void main(String[] args) {

        String datasetRoot = args[0];
        DataSource ds = new DataSource(datasetRoot);

        SparkSession spark = SparkSession.builder().appName("csv-by-sql-demo").getOrCreate();
        Map<String, String> yelpTBLName = ds.getYELPDatasets();

         Dataset<Row> business = spark.read().json(yelpTBLName.get("business"));
         Dataset<Row> review = spark.read().json(yelpTBLName.get("review"));
         Dataset<Row> user = spark.read().json(yelpTBLName.get("user"));
         Dataset<Row> checkin = spark.read().json(yelpTBLName.get("checkin"));
         Dataset<Row> tip = spark.read().json(yelpTBLName.get("tip"));
         Dataset<Row> photo = spark.read().json(yelpTBLName.get("photo"));

         Dataset<Row> df = mergeDataFrames(business, review, "business_id");
         //df = mergeDataFrames(df, checkin, "business_id");
         //df = mergeDataFrames(df, tip, "business_id");
         df = mergeDataFrames(df, user, "user_id");
         //df = mergeDataFrames(df, photo, "business_id");
    }

    public static Dataset<Row> mergeDataFrames(Dataset<Row> df1, Dataset<Row> df2, String col_name){
        return df1.join(df2, col_name, "left");
    }
}
