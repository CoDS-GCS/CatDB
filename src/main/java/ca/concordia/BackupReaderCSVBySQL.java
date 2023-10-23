package ca.concordia;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;

public class BckupReaderCSVBySQL {

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

        Dataset<Row> business2 = business.select("address", "business_id", "categories", "city",
                "is_open", "latitude", "longitude", "name", "postal_code", "review_count", "stars", "state");
//
//        Dataset<Row> df = mergeDataFrames(business, review, "business_id");
        //df = mergeDataFrames(df, checkin, "business_id");
        //df = mergeDataFrames(df, tip, "business_id");
        //df = mergeDataFrames(df, user, "user_id");
        //df = mergeDataFrames(df, photo, "business_id");
        //business.printSchema();
        //review.collectAsList();
        //System.out.println("++++++++++++++++++++++++++++++++++++++++");
       // System.out.println(">>>>>>>>>>>>>>>>>> "+review.count());

        business2.printSchema();
        business2.show();

        business2.repartition(1).write().option("header", "true").csv("/home/saeed/YELPCSV/business.csv");
        //review.repartition(1).write().option("header", "true").csv("/home/saeed/YELPCSV/review.csv");
        //user.repartition(1).write().option("header", "true").csv("/home/saeed/YELPCSV/user.csv");
        //checkin.repartition(1).write().option("header", "true").csv("/home/saeed/YELPCSV/checkin.csv");
        //tip.repartition(1).write().option("header", "true").csv("/home/saeed/YELPCSV/tip.csv");
        //photo.repartition(1).write().option("header", "true").csv("/home/saeed/YELPCSV/photo.csv");

        //business.select("attributes.*").repartition(1).write().csv("/home/saeed/YELPCSV/attributes.csv");

    }

    public static Dataset<Row> mergeDataFrames(Dataset<Row> df1, Dataset<Row> df2, String col_name){
        return df1.join(df2, col_name, "left");
    }
}
