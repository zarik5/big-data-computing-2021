import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;


public class G23HW2 {
    public static Tuple2<Vector, Integer> strToTuple (String str){
        String[] tokens = str.split(",");
        // Here should be length -1, because the last element should be the index
        double[] data = new double[tokens.length - 1];
        for (int i = 0; i < tokens.length-1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        Vector point = (Vector) Vectors.dense(data);
        Integer cluster = Integer.valueOf(tokens[tokens.length-1]);
        Tuple2<Vector, Integer> pair = new Tuple2<>(point, cluster);
        return pair;
    }

    public static void main(String[] args) throws IllegalArgumentException {
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path num_clusters sample_size");
        }

        // Spark setup
        JavaSparkContext context = new JavaSparkContext(new SparkConf(true).setAppName("G23HW2"));

        // Input file path
        String filePath = args[0];
        // Clusters count
        int k = Integer.parseInt(args[1]);
        // Sample size per cluster
        int t = Integer.parseInt(args[2]);
        System.out.println("INPUT PARAMETERS: file=" + filePath + " k=" + k + " t=" + t + "\n");

        // =========================== STEP 1: Read the input data ===========================
        JavaPairRDD<Vector,Integer> fullClustering = context.textFile(filePath).repartition(k)
                .mapToPair(x -> strToTuple(x));


        // ===================== STEP 2:Computed the sharedClusterSizes =================================================
        List<Long> sharedClusterSizes = new ArrayList<Long>(fullClustering.map(data->(data._2())).countByValue().values());

        // ============ STEP 3: Extract a sample of the input clustering, which named clusteringSample ===================
        // each point is selected independently with probability min{t/|C|, 1}
        // t: sample size
        // |C| size of the cluster C
        // k: cluster count
        ArrayList<Vector> clusteringSample;
        clusteringSample = new ArrayList<>();
        clusteringSample.addAll(fullClustering
                // use random() to decide whether the element should be chosen
                .filter(data-> {
                    double p = Math.random();
                    if(p < Math.min((double)t / (double)sharedClusterSizes.get(data._2()), 1 )) {
                        return true;
                    }
                    return false;
                }).map(data-> {
                    return data._1();
                }).collect());

    }
}
