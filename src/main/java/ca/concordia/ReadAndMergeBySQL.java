package ca.concordia;

import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public class ReadAndMergeBySQL {

    public static void main(String[] args) {

        String datasetRoot = args[0];
        String query = args[1];

        DataSource ds = new DataSource(datasetRoot);

        SparkSession spark = SparkSession.builder().appName("csv-by-sql-demo").getOrCreate();
        Map<String, String> yelpTBLName = ds.getYELPDatasets();

         Dataset<Row> business = spark.read().json(yelpTBLName.get("business"));
         Dataset<Row> review = spark.read().json(yelpTBLName.get("review"));
         Dataset<Row> user = spark.read().json(yelpTBLName.get("user"));
         String queryString = "";

         switch (query){
             case "Q1": queryString = "select b.name, b.city, b.stars as business_stars,\n" +
                            "           u.name as user_name, u.average_stars, u.yelping_since,\n" +
                            "           r.date, r.funny, r.stars, r.text\n" +
                            "           from business b inner join review r on b.business_id= r.business_id\n" +
                            "                inner join users u on u.user_id = r.user_id"; break;

             case "Q2": queryString ="select b.name, b.city, b.stars as business_stars,\n" +
                     "                                u.name as user_name, u.average_stars, u.yelping_since,\n" +
                     "                                r.date, r.funny, r.stars, r.text\n" +
                     "                         from business b inner join review r on b.business_id= r.business_id\n" +
                     "                                         inner join users u on u.user_id = r.user_id\n" +
                     "                         where b.stars <=5"; break;

             case "Q3": queryString = "select b.name, b.city, b.stars as business_stars,\n" +
                     "                                u.name as user_name, u.average_stars, u.yelping_since,\n" +
                     "                                r.date, r.funny, r.stars, r.text\n" +
                     "                         from business b inner join review r on b.business_id= r.business_id\n" +
                     "                                         inner join users u on u.user_id = r.user_id\n" +
                     "                         where r.stars =2 "; break;

             case "Q4": queryString = "select b.name, b.city, b.stars as business_stars,\n" +
                     "                                u.name as user_name, u.average_star, u.yelping_since,\n" +
                     "                                r.date, r.funny, r.stars, r.text\n" +
                     "                         from business b inner join review r on b.business_id= r.business_id\n" +
                     "                                         inner join users u on u.user_id = r.user_id\n" +
                     "                         where r.stars =2 or r.stars=5"; break;
             case "Q5": queryString = "select b.name, b.city, b.stars as business_stars,\n" +
                     "                                u.name as user_name, u.average_stars, u.yelping_since,\n" +
                     "                                r.date, r.funny, r.stars, r.text\n" +
                     "                         from (select business_id, user_id, date, funny, stars, text from review where stars =2) r inner join\n" +
                     "                             business b  on b.business_id= r.business_id\n" +
                     "                                         inner join users u on u.user_id = r.user_id"; break;

         }
        Dataset<Row> sqlDF = runQuery(queryString, business, review, user, spark);

         AtomicLong count = new AtomicLong();
        sqlDF.foreach((ForeachFunction<Row>) row -> count.getAndIncrement());
        System.out.println(">>>>>>>>>>>> COUNT="+count.get());



    }

    static  Dataset<Row> runQuery(String query_string,Dataset<Row> business, Dataset<Row> review, Dataset<Row> user, SparkSession spark){
        business.createOrReplaceTempView("business");
        review.createOrReplaceTempView("review");
        user.createOrReplaceTempView("users");
        Dataset<Row> sqlDF = spark.sql(query_string);

        return sqlDF;
    }
}
