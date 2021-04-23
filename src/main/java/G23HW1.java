import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class G23HW1 {
    public static void main(String[] args) throws IllegalArgumentException {
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: num_partitions num_results file_path");
        }

        // Spark setup
        JavaSparkContext context = new JavaSparkContext(new SparkConf(true).setAppName("G23HW1"));

        // Partitions count
        int K = Integer.parseInt(args[0]);
        // Number of results
        int T = Integer.parseInt(args[1]);
        // Input file path
        String filePath = args[2];
        System.out.println("INPUT PARAMETERS: K=" + K + " T=" + T + " file=" + filePath + "\n");

        // =========================== STEP 1: Split document into lines ===========================
        JavaRDD<String> RawData = context.textFile(filePath).repartition(K); //.cache();

        // ================= STEP 2: Get normalizedRatings (ProductID, NormRating) =================
        JavaPairRDD<String, Float> normalizedRatings = RawData
                // MAP -> (UserID, (ProductID, Rating))
                .mapToPair(line -> {
                    // extract data from document line
                    String[] tokens = line.split(",");
                    String userID = tokens[1];
                    String productID = tokens[0];
                    float rating = Float.parseFloat(tokens[2]);
                    return new Tuple2<>(userID, new Tuple2<>(productID, rating));
                })
                // REDUCE -> (ProductID, NormRating)
                // Analysis: In the worst case (one user that writes all reviews), groupByKey() will
                // collect all data in one worker, leading to ML=O(N). In practice this should not
                // be the case, one user will write less than sqrt(N) reviews.
                .groupByKey()
                .flatMapToPair(ratingsByUser -> {
                    // calculate average
                    float average = (float) StreamSupport
                            .stream(ratingsByUser._2().spliterator(), true)
                            .mapToDouble(Tuple2::_2)
                            .average()
                            .orElse(0);

                    // reconstruct pairs
                    return StreamSupport
                            .stream(ratingsByUser._2().spliterator(), true)
                            .map(pair -> new Tuple2<>(pair._1(), pair._2() - average))
                            .iterator();
                });

        // ====================== STEP 3: Get maxNormRatings (ProductID, MNR) ======================
        JavaPairRDD<String, Float> maxNormRatings = normalizedRatings
                // ROUND 1: REDUCE -> (ProductID, PartialMNR)
                .mapPartitionsToPair(normRatings -> {
                    // accumulate partial max ratings
                    HashMap<String, Float> partialMNR = new HashMap<>();
                    while (normRatings.hasNext()) {
                        Tuple2<String, Float> pair = normRatings.next();
                        partialMNR.compute(pair._1(), (__, value) ->
                                value != null ? Math.max(value, pair._2()) : pair._2()
                        );
                    }

                    // reconstruct pairs
                    return partialMNR
                            .entrySet()
                            .stream()
                            .map(entry -> new Tuple2<>(entry.getKey(), entry.getValue()))
                            .iterator();
                })
                // ROUND 2: REDUCE -> (ProductID, MNR)
                .reduceByKey(Math::max);

        // =============================== STEP 4: Get top T ratings ===============================
        List<Tuple2<String, Float>> topRatings = maxNormRatings
                // MAP -> (AscendingRank, (ProductID, MNR))
                .mapToPair(pair -> new Tuple2<>(-pair._2(), pair))
                // REDUCE -> (AscendingRank, (ProductID, MNR))
                .sortByKey()
                .take(T)
                .stream()
                .map(Tuple2::_2)
                .collect(Collectors.toList());

        System.out.println("OUTPUT:");
        for (Tuple2<String, Float> entry : topRatings) {
            System.out.println("Product " + entry._1() + " maxNormRating " + entry._2());
        }
    }
}
