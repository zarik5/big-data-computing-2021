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
import org.apache.spark.sql.sources.In;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class G23HW3 {

    // Note: "clustering" can be the full clustering or a sample
    // normalization_factors depends on whether the full or approximated clustering is requested
    static float clustering_silhouette_for_point(
            List<Tuple2<Vector, Integer>> clustering,
            Vector point,
            int cluster_idx,
            Map<Integer, Long> normalization_factors
    ) {
        float a = (float) clustering
                .stream()
                .sequential()
                .filter(pair -> pair._2 == cluster_idx)
                .mapToDouble(pair -> Vectors.sqdist(point, pair._1))
                .sum() / normalization_factors.get(cluster_idx);

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
                .mapToDouble(entry -> entry.getValue() / normalization_factors.get(entry.getKey()))
                .min()
                .orElse(0);

        return (b - a) / Math.max(a, b);
    }

    public static void main(String[] args) throws IllegalArgumentException {
        if (args.length != 6) {
            throw new IllegalArgumentException("USAGE: file_path num_clusters sample_size");
        }

        // Spark setup
        JavaSparkContext context =
                new JavaSparkContext(new SparkConf(true)
                        .setAppName("G23HW3")
                        .set("spark.locality.wait", "0s"));

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


        System.out.println("INPUT PARAMETERS: file=" + filePath + " kstart=" + kstart + " iter=" + iter + " M=" + M + " L=" + L + "\n");

        // ============================== STEP 1: Read the input data ==============================
        long start_time_ms = System.currentTimeMillis();

        JavaRDD<Vector> inputPoints = context.textFile(filePath).repartition(L).map(s -> {
            String[] sarry = s.split(" ");
            double[] values = new double[sarry.length];
            for (int i = 0; i < sarry.length; i++) {
                values[i] = Double.parseDouble(sarry[i]);
            }
            return Vectors.dense(values);
        }).cache();

        long input_time_ms = System.currentTimeMillis() - start_time_ms;

        System.out.println("Time for input reading = " + input_time_ms);

        for(int k = kstart; k < kstart + h; k++) {
            //System.out.println("\n" + "Number of clusters k = " + k);

            // Computes a clustering of the input points with k clusters
            start_time_ms = System.currentTimeMillis();

            KMeansModel clusters = KMeans.train(inputPoints.rdd(), k, iter);

            // RDD currentClustering of pairs (point, cluster_index)
            JavaPairRDD<Vector, Integer>currentClustering = inputPoints.repartition(L).mapToPair(data -> {
                return new Tuple2<>(data, clusters.predict(data));
            }).cache();

            long clustering_time_ms = System.currentTimeMillis() - start_time_ms;

            // Use the code in HW2
            Integer t = M / k;
            // ==========================  Compute sharedClusterSizes ===========================
            // A map is used instead of a list or array, to preserve the cluster index (in case there
            // are skips in the indices)
            Broadcast<Map<Integer, Long>> sharedClusterSizes =
                    context.broadcast(currentClustering.map(Tuple2::_2).countByValue());

            start_time_ms = System.currentTimeMillis();
            // ============================= Get clusteringSample ==============================
            // Note: Especially in this step, storing the points as (cluster_idx, vector) would have
            // been more ergonomic than (vector, cluster_idx), saving two "mapToPair()".
            Broadcast<List<Tuple2<Vector, Integer>>> clusteringSample = context.broadcast(
                    currentClustering
                            // Poisson sampling (P[x < min{t/|C|, 1}])
                            .filter(data -> Math.random() <
                                    Math.min((float) t / sharedClusterSizes.value().get(data._2), 1))
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



            // ================ Calculate exact silhouette on clusteringSample =================

            Map<Integer, Long> cluster_sample_sizes = new HashMap<>();
            for (Tuple2<Vector, Integer> pair : clusteringSample.value()) {
                cluster_sample_sizes.compute(pair._2, (__, sum) -> (sum != null ? sum : 0) + 1);
            }

            float exactSilhSample = clusteringSample
                    .value()
                    .stream()
                    .sequential()
                    .map(point -> G23HW3.clustering_silhouette_for_point(
                            clusteringSample.value(),
                            point._1,
                            point._2,
                            cluster_sample_sizes)
                    )
                    .reduce(Float::sum)
                    .orElse(0f) / clusteringSample.value().size();

            long exact_sample_computation_time_ms = System.currentTimeMillis() - start_time_ms;

            // Number of clusters k
            System.out.println("\n" + "Number of clusters k = " + k);
            // Silhouette coefficient
            System.out.println("Silhouette coefficient = " + exactSilhSample);
            // Time for clustering
            System.out.println("Time for clustering = " + clustering_time_ms);
            // Time for silhouette computation
            System.out.println("Time for silhouette computation = " + exact_sample_computation_time_ms);
        }



    }
}
