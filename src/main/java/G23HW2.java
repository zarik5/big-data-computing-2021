import java.util.*;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

public class G23HW2 {
    static Tuple2<Vector, Integer> strToTuple(String str) {
        String[] tokens = str.split(",");
        // Here should be length - 1, because the last element should be the index
        double[] data = new double[tokens.length - 1];
        for (int i = 0; i < tokens.length - 1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        Vector point = Vectors.dense(data);
        Integer cluster = Integer.valueOf(tokens[tokens.length - 1]);
        return new Tuple2<>(point, cluster);
    }

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
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path num_clusters sample_size");
        }

        // Spark setup
        JavaSparkContext context =
                new JavaSparkContext(new SparkConf(true).setAppName("G23HW2"));

        // Input file path
        String filePath = args[0];
        // Clusters count (not actually used)
        int k = Integer.parseInt(args[1]);
        // Sample size per cluster
        int t = Integer.parseInt(args[2]);
        System.out.println("INPUT PARAMETERS: file=" + filePath + " k=" + k + " t=" + t + "\n");

        // ============================== STEP 1: Read the input data ==============================
        final int PARTITIONS = 8;
        JavaPairRDD<Vector, Integer> fullClustering =
                context.textFile(filePath).repartition(PARTITIONS).mapToPair(G23HW2::strToTuple);

        // ========================== STEP 2: Compute sharedClusterSizes ===========================
        // A map is used instead of a list or array, to preserve the cluster index (in case there
        // are skips in the indices)
        Broadcast<Map<Integer, Long>> sharedClusterSizes =
                context.broadcast(fullClustering.map(Tuple2::_2).countByValue());

        // ============================= STEP 3: Get clusteringSample ==============================
        // Note: Especially in this step, storing the points as (cluster_idx, vector) would have
        // been more ergonomic than (vector, cluster_idx), saving two "mapToPair()".
        Broadcast<List<Tuple2<Vector, Integer>>> clusteringSample = context.broadcast(
                fullClustering
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

        // ============== STEP 4: Calculate approximate silhouette on fullClustering ===============
        long start_time_ms = System.currentTimeMillis();

        // normalization_factors is computed sequentially because sharedClusterSizes is small
        // enough.
        Broadcast<Map<Integer, Long>> normalization_factors = context.broadcast(new HashMap<>());
        for (Map.Entry<Integer, Long> entry : sharedClusterSizes.value().entrySet()) {
            normalization_factors.value().compute(entry.getKey(),
                    (__, sum) -> (sum != null ? sum : 0) + Math.min(t, entry.getValue())
            );
        }

        float approxSilhFull = fullClustering
                .map(point -> G23HW2.clustering_silhouette_for_point(
                        clusteringSample.value(),
                        point._1,
                        point._2,
                        normalization_factors.value())
                )
                .reduce(Float::sum) / fullClustering.count();

        long approx_fill_computation_time_ms = System.currentTimeMillis() - start_time_ms;

        // ================ STEP 5: Calculate exact silhouette on clusteringSample =================
        start_time_ms = System.currentTimeMillis();

        Map<Integer, Long> cluster_sample_sizes = new HashMap<>();
        for (Tuple2<Vector, Integer> pair : clusteringSample.value()) {
            cluster_sample_sizes.compute(pair._2, (__, sum) -> (sum != null ? sum : 0) + 1);
        }

        float exactSilhSample = clusteringSample
                .value()
                .stream()
                .sequential()
                .map(point -> G23HW2.clustering_silhouette_for_point(
                        clusteringSample.value(),
                        point._1,
                        point._2,
                        cluster_sample_sizes)
                )
                .reduce(Float::sum)
                .orElse(0f) / clusteringSample.value().size();

        long exact_sample_computation_time_ms = System.currentTimeMillis() - start_time_ms;

        // ================================ STEP 6: Display results ================================
        System.out.println("OUTPUT:");
        System.out.println("Value of approxSilhFull = " + approxSilhFull);
        System.out.println("Time to compute approxSilhFull = "
                + approx_fill_computation_time_ms + " ms");
        System.out.println("Value of exactSilhSample = " + exactSilhSample);
        System.out.println("Time to compute exactSilhSample = "
                + exact_sample_computation_time_ms + " ms");

    }
}
