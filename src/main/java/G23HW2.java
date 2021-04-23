import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;


public class G23HW2 {
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
    }
}
