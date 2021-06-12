import com.google.common.collect.Iterables;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class G23HW3 {

    // Note: "clustering" can be the full clustering or a sample
    // normalization_factors depends on whether the full or approximated clustering is requested
    static float clustering_silhouette_for_point(
            List<Tuple2<Vector, Integer>> clustering,
            Vector point,
            int cluster_idx,
            List<Broadcast<Long>> normalization_factors
    ) {
        float a = (float) clustering
                .stream()
                .sequential()
                .filter(pair -> pair._2 == cluster_idx)
                .mapToDouble(pair -> Vectors.sqdist(point, pair._1))
                .sum() / normalization_factors.get(cluster_idx).value();

        List<Tuple2<Double, Integer>> distances = clustering
                .stream()
                .sequential()
                .filter(pair -> pair._2 != cluster_idx)
                .map(pair -> new Tuple2<>(Vectors.sqdist(point, pair._1), pair._2))
                .collect(Collectors.toList());
        Map<Integer, Double> sums = new HashMap<>();
        for (Tuple2<Double, Integer> pair : distances) {
            sums.compute(pair._2, (__, sum) -> (sum != null ? sum : 0) + pair._1);
        }
        float b = (float) sums
                .entrySet()
                .stream()
                .sequential()
                .mapToDouble(entry -> 
                        entry.getValue() / normalization_factors.get(entry.getKey()).value()
                )
                .min()
                .orElse(0);

        return (b - a) / Math.max(a, b);
    }

    public static void main(String[] args) throws IllegalArgumentException {
        if (args.length != 6) {
            throw new IllegalArgumentException("USAGE: file_path kstart h iter M L");
        }

        // Spark setup
        JavaSparkContext context =
                new JavaSparkContext(new SparkConf(true)
                        .setAppName("G23HW3")
                        .set("spark.locality.wait", "0s"));
        context.setLogLevel("WARN");

        // Input file path
        String filePath = args[0];
        // The initial number of clusters
        int kstart = Integer.parseInt(args[1]);
        // The number of values of k that the program will test
        int h = Integer.parseInt(args[2]);
        // The number of iterations of Lloyd's algorithm
        int iter = Integer.parseInt(args[3]);
        // The expected size of the sample used to approximate the silhouette coefficient
        int M = Integer.parseInt(args[4]);
        // The number of partitions of the RDDs containing the input points and their clustering
        int L = Integer.parseInt(args[5]);


        System.out.println("INPUT PARAMETERS: file=" + filePath + " kstart=" + kstart + " h=" + h +
                " iter=" + iter + " M=" + M + " L=" + L + "\n");

        // ================================== Read the input data ==================================
        long start_time_ms = System.currentTimeMillis();

        JavaRDD<Vector> inputPoints = context.textFile(filePath).repartition(L).map(s -> {
            String[] arr = s.split(" ");
            double[] values = new double[arr.length];
            for (int i = 0; i < arr.length; i++) {
                values[i] = Double.parseDouble(arr[i]);
            }
            return Vectors.dense(values);
        }).cache();

        long input_time_ms = System.currentTimeMillis() - start_time_ms;

        System.out.println("Time for input reading = " + input_time_ms);

        for (int k = kstart; k < kstart + h; k++) {
            // ================================ Compute clustering =================================
            start_time_ms = System.currentTimeMillis();

            KMeansModel centers = KMeans.train(inputPoints.rdd(), k, iter);

            // RDD currentClustering of pairs (point, cluster_index)
            JavaPairRDD<Vector, Integer> currentClustering =
                    inputPoints.mapToPair(point -> new Tuple2<>(point,
                            centers.predict(point))).cache();

            long clustering_time_ms = System.currentTimeMillis() - start_time_ms;

            // ================= Compute approximate average silhouette coefficient ================
            start_time_ms = System.currentTimeMillis();
            int t = M / k;

            Broadcast<Map<Integer, Long>> sharedClusterSizes =
                    context.broadcast(currentClustering.map(Tuple2::_2).countByValue());

            Broadcast<List<Tuple2<Vector, Integer>>> clusteringSample = context.broadcast(
                    currentClustering
                            // Poisson sampling (P[x < min{t/|C|, 1}])
                            .filter(data -> Math.random() <
                                    Math.min((float) t / sharedClusterSizes.value().get(data._2), 1)
                            )
                            .mapToPair(pair -> new Tuple2<>(pair._2, pair._1))
                            // Group by cluster
                            .groupByKey()
                            // Take at most t points per cluster
                            .mapValues(iterator -> Iterables.limit(iterator, t))
                            // Flatten
                            .flatMapValues(iterable -> iterable)
                            .mapToPair(pair -> new Tuple2<>(pair._2, pair._1))
                            .collect()
            );

            // normalization_factors is computed sequentially because sharedClusterSizes is small
            // enough.
            Map<Integer, Long> normalization_factors_temp = new HashMap<>();
            for (Map.Entry<Integer, Long> entry : sharedClusterSizes.value().entrySet()) {
                normalization_factors_temp.compute(entry.getKey(),
                        (__, sum) -> (sum != null ? sum : 0) + Math.min(t, entry.getValue())
                );
            }

            // This is a patch added for the homework 3. Long values were null when transferred to
            // workers if the they are not wrapped in Broadcast.
            Broadcast<List<Broadcast<Long>>> normalization_factors = context.broadcast(
                    normalization_factors_temp
                            .values()
                            .stream()
                            .map(context::broadcast)
                            .collect(Collectors.toList())
            );

            float approxSilhFull = currentClustering
                    .map(point -> G23HW3.clustering_silhouette_for_point(
                            clusteringSample.value(),
                            point._1,
                            point._2,
                            normalization_factors.value())
                    )
                    .reduce(Float::sum) / currentClustering.count();

            long approx_full_computation_time_ms = System.currentTimeMillis() - start_time_ms;

            // ================================== Display results ==================================
            // Number of clusters k
            System.out.println("\n" + "Number of clusters k = " + k);
            // Silhouette coefficient
            System.out.println("Silhouette coefficient = " + approxSilhFull);
            // Time for clustering
            System.out.println("Time for clustering = " + clustering_time_ms);
            // Time for silhouette computation
            System.out.println("Time for silhouette computation = "
                    + approx_full_computation_time_ms);
        }
    }
}
