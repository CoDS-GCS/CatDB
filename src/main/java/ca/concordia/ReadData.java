package ca.concordia;

import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public class ReadData {

    public static void main(String[] args) {

        String datasetRoot = args[0];
        String query = args[1];

        DataSource ds = new DataSource(datasetRoot);

        SparkSession spark = SparkSession.builder().appName("csv-by-sql-demo").getOrCreate();
        Map<String, String> yelpTBLName = ds.getYELPDatasets();

        Dataset<Row> business = spark.read().json(yelpTBLName.get("business"));
        Dataset<Row> review = spark.read().json(yelpTBLName.get("review"));
        Dataset<Row> user = spark.read().json(yelpTBLName.get("user"));

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
