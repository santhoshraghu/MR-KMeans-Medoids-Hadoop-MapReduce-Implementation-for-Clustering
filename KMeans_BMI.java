import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMeans_BMI {
    public static int centroidCount = 0;
    public static List<ClusterCenter> centroids = new ArrayList<ClusterCenter>();
    public static List<ClusterCenter> previousCentroids = new ArrayList<ClusterCenter>();
    public static List<Point> k_seeds = new ArrayList<Point>();

    public static class Point implements Writable {
        private double x, y;

        public Point() {
            x = 0.0;
            y = 0.0;
        }

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            dataOutput.writeDouble(x);
            dataOutput.writeDouble(y);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            x = dataInput.readDouble();
            y = dataInput.readDouble();
        }
    }

    public static class ClusterCenter implements Writable {
        public Point point = new Point(0, 0);
        public int id;
        public int count = 0;

        public List<Point> points;

        public ClusterCenter() {
            this.points = new ArrayList<Point>();
        }

        public ClusterCenter(int id, Point point) {
            this.points = new ArrayList<Point>();
            this.id = id;
            this.point = point;
        }

        public ClusterCenter(Point point) {
            this.point = point;
        }

        public Point getPoint() {
            return point;
        }

        public int get() {
            return count;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            point.write(dataOutput);
            dataOutput.writeInt(count);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            point.readFields(dataInput);
            count = dataInput.readInt();
        }

        public void addPoint(Point point) {
            points.add(point);
        }
    }

    public static class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Mapper logic: Assign data points to the nearest centroid
            String[] pointData = value.toString().split(",");
            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);
            Point point = new Point(x, y);
            ClusterCenter nearestCenter = findNearestCluster(point);
            nearestCenter.addPoint(point);
            context.write(new IntWritable(nearestCenter.id), new Text(x + ":" + y));
        }

        private ClusterCenter findNearestCluster(Point point) {
            ClusterCenter nearestCenter = null;
            double temp = Double.MAX_VALUE;
            for (ClusterCenter center : centroids) {
                Point centerPoint = center.getPoint();
                double distance = calculateEuclideanDistance(point, centerPoint);
                if (distance < temp) {
                    nearestCenter = center;
                }
                temp = distance;
            }
            return nearestCenter;
        }

        private double calculateEuclideanDistance(Point p1, Point p2) {
            double dx = p1.getX() - p2.getX();
            double dy = p1.getY() - p2.getY();
            return Math.sqrt(dx * dx + dy * dy);
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Reducer logic: Recalculate cluster centers based on assigned data points
            List<Point> points = new ArrayList<>();
            for (Text value : values) {
                String[] parts = value.toString().split(":");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                points.add(new Point(x, y));
            }
            ClusterCenter newCenter = computeNewCenter(points);
            context.write(key, new Text(newCenter.point.getX() + "," + newCenter.point.getY()));
        }

        private ClusterCenter computeNewCenter(List<Point> points) {
            double totalX = 0.0;
            double totalY = 0.0;
            for (Point point : points) {
                totalX += point.getX();
                totalY += point.getY();
            }
            double avgX = totalX / points.size();
            double avgY = totalY / points.size();
            return new ClusterCenter(new Point(avgX, avgY));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMeans");

        String inputPath, kSeedData, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        kSeedData = "C://Users//user//Downloads//seed_points.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_2b_k3";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 3; // Set the desired value of k

        // Read all points from the seed points file and store them in a list
        List<ClusterCenter> seedPoints = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(kSeedData))) {
            String line;
            int j = 0;
            while (j < k && (line = br.readLine()) != null) { // Read the first k lines
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                seedPoints.add(new ClusterCenter(j + 1, new Point(x, y)));
                j++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Set the centroidCount to k
        centroidCount = k;
        System.out.println("All seed points" + seedPoints);
        System.out.println("K value: " + centroidCount);

        List<ClusterCenter> selectedCentroids = getRandomCentroids(seedPoints, k);

        // Initialize your centroids list with the selected random k points
        centroids = selectedCentroids;
        previousCentroids = new ArrayList<>(centroids); // Store the previous centroids

        job.setJarByClass(KMeans_BMI.class);
        job.setMapperClass(KMeansMapper.class);
        job.setReducerClass(KMeansReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        int maxIterations = 10;
        boolean converged = false; // To check if centroids have not changed for two consecutive iterations

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            boolean jobResult = job.waitForCompletion(true);
            if (!jobResult) {
                System.err.println("KMeans Job Failed");
                System.exit(1);
            }

            // Update centroids for the next iteration
            if (iteration < maxIterations) {
                previousCentroids.clear();
                previousCentroids.addAll(centroids);
                centroids.clear();

                // Calculate new centroids based on the reducer's output
                for (int i = 0; i < k; i++) {
                    List<Point> clusterPoints = new ArrayList<>();
                    for (ClusterCenter center : previousCentroids) { // Use previousCentroids here
                        if (center.id == i) {
                            clusterPoints.addAll(center.points);
                        }
                    }

                    if (!clusterPoints.isEmpty()) {
                        // Calculate the average of the cluster points to get the new centroid
                        double totalX = 0.0;
                        double totalY = 0.0;
                        for (Point point : clusterPoints) {
                            totalX += point.getX();
                            totalY += point.getY();
                        }

                        double avgX = totalX / clusterPoints.size();
                        double avgY = totalY / clusterPoints.size();

                        centroids.add(new ClusterCenter(i, new Point(avgX, avgY)));
                    } else {
                        // If the cluster is empty, keep the previous centroid
                        centroids.add(previousCentroids.get(i));
                    }
                }

                // Check for convergence
                boolean hasConverged = true;
                for (int i = 0; i < k; i++) {
                    double distance = calculateEuclideanDistance(centroids.get(i).point, previousCentroids.get(i).point);
                    if (distance > 0.0) {
                        hasConverged = false;
                        break;
                    }
                }

                if (hasConverged) {
                    System.out.println("Converged after " + iteration + " iterations.");
                    converged = true; // Set the converged flag if centroids have converged
                    break; // Terminate the loop
                }
            }
        }
    }

    private static List<ClusterCenter> getRandomCentroids(List<ClusterCenter> seedPoints, int k) {
        List<ClusterCenter> selectedCentroids = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis()); // Use current time as the seed

        // Ensure k is not greater than the number of available seed points
        if (k > seedPoints.size()) {
            k = seedPoints.size();
        }

        while (selectedCentroids.size() < k) {
            int randomIndex = rand.nextInt(seedPoints.size());
            ClusterCenter randomPoint = seedPoints.get(randomIndex);

            // Avoid selecting the same point multiple times
            if (!selectedCentroids.contains(randomPoint)) {
                selectedCentroids.add(randomPoint);
            }
        }

        return selectedCentroids;
    }

    private static double calculateEuclideanDistance(Point p1, Point p2) {
        double dx = p1.getX() - p2.getX();
        double dy = p1.getY() - p2.getY();
        return Math.sqrt(dx * dx + dy * dy);
    }
}
