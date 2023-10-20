package ca.concordia;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ReaderCSV {

    public static void main(String[] args) {

        String dataset = args[0];
        System.out.println("+++++++++++++++++++++  "+ dataset);

        // TODO Auto-generated method stub
        SparkSession spark = SparkSession.builder().appName("csv-demo").getOrCreate();
        Dataset<Row> employees = spark.read().csv("/home/saeed/Documents/Github/CatDB/Experiments/data/IMDB/ImdbName.csv");
        employees.printSchema();
        employees.show();
        //employees.write().json("file:///home/rahul/Desktop/eclipse-ee/employees_spark.json");
    }
}
