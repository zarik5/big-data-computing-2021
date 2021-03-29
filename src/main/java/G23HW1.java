import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Product;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class G23HW1 {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // spark setup
        SparkConf conf = new SparkConf(true).setAppName("HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // STEP 1
        int K = Integer.parseInt(args[0]);
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(K).cache();

        // STEP 2
        // (UserID, (ProductID, Rating))
        JavaPairRDD<String, Tuple2<String, Float>> userRating;
        userRating = docs.flatMapToPair((doc) -> {
            String[] tokens = doc.split(",");
            ArrayList<Tuple2<String, Tuple2<String, Float>>> pairs = new ArrayList<>();
            String userID = tokens[1];
            String productID = tokens[0];
            Float rating = Float.parseFloat(tokens[2]);
            pairs.add(new Tuple2<>(userID, new Tuple2<>(productID, rating)));
            return pairs.iterator();
        });

        // (ProductID, (Rating, avgRating)
        JavaPairRDD<String, Tuple2<Float, Float>> avgRatings = userRating.groupByKey().flatMapToPair(info -> {
            float sum = 0;
            float count = 0;
            ArrayList<Tuple2<String, Tuple2<Float, Float>>> pairs = new ArrayList<>();
            Iterator<Tuple2<String, Float>> it = info._2().iterator();
            while (it.hasNext()) {
                sum += it.next()._2();
                count++;
            }
            float ave = sum / count;
            Iterator<Tuple2<String, Float>> it2 = info._2().iterator();
            while (it2.hasNext()) {
                Tuple2<String, Float> tmp = it2.next();
                pairs.add(new Tuple2<>(tmp._1(), new  Tuple2<>(tmp._2(), ave)));
            }
            return pairs.iterator();
        });

//        avgRatings.foreach(data -> {
//            System.out.println(data);
//        });

        // normalizedRatings (ProductID, NormalRating)
        JavaPairRDD<String , Float> normalizedRating = avgRatings.mapToPair(doc -> {
            return new Tuple2<>(doc._1(), doc._2()._1() - doc._2()._2());
        });

//        normalizedRating.foreach(data -> {
//            System.out.println(data);
//        });

        // STEP 3


        // STEP 4

    }
}
