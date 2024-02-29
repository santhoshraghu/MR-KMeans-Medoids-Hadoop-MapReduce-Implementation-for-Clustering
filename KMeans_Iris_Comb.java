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

public class KMeans_Iris_Comb {
    public static int centroidCount = 0;
    public static List<ClusterCenter> centroids = new ArrayList<ClusterCenter>();
    public static List<ClusterCenter> previousCentroids = new ArrayList<ClusterCenter>();
    public static final double EARLY_CONVERGENCE_THRESHOLD = 0.001;

    public static class Point implements Writable {
        private double x, y, z, w;

        public Point() {
            x = 0.0;
            y = 0.0;
            z = 0.0;
            w = 0.0;
        }

        public Point(double x, double y, double z, double w) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        public double getZ() {
            return z;
        }

        public double getW() {
            return w;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            dataOutput.writeDouble(x);
            dataOutput.writeDouble(y);
            dataOutput.writeDouble(z);
            dataOutput.writeDouble(w);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            x = dataInput.readDouble();
            y = dataInput.readDouble();
            z = dataInput.readDouble();
            w = dataInput.readDouble();
        }
    }

    public static class ClusterCenter implements Writable {
        public Point point = new Point(0, 0, 0, 0);
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

    public static class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, Point> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] pointData = value.toString().split(",");
            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);
            double z = Double.parseDouble(pointData[2]);
            double w = Double.parseDouble(pointData[3]);
            Point point = new Point(x, y, z, w);
            ClusterCenter nearestCenter = findNearestCluster(point);
            nearestCenter.addPoint(point);
            context.write(new IntWritable(nearestCenter.id), point);
        }

        private ClusterCenter findNearestCluster(Point point) {
            ClusterCenter nearestCenter = null;
            double temp = Double.MAX_VALUE;
            for (ClusterCenter center : centroids) {
                Point centerPoint = center.getPoint();
                double distance = calculateEuclideanDistance(point, centerPoint);
                if (distance < temp) {
                    nearestCenter = center;
                    temp = distance;
                }
            }
            return nearestCenter;
        }

        private double calculateEuclideanDistance(Point p1, Point p2) {
            double dx = p1.getX() - p2.getX();
            double dy = p1.getY() - p2.getY();
            double dz = p1.getZ() - p2.getZ();
            double dw = p1.getW() - p2.getW();
            return Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
        }
    }

    public static class KMeansCombiner extends Reducer<IntWritable, Point, IntWritable, Point> {
        @Override
        public void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Point point : values) {
                points.add(new Point(point.getX(), point.getY(), point.getZ(), point.getW()));
            }
            ClusterCenter newCenter = computeNewCenter(points);
            context.write(key, newCenter.getPoint());
        }

        private ClusterCenter computeNewCenter(List<Point> points) {
            double totalX = 0.0;
            double totalY = 0.0;
            double totalZ = 0.0;
            double totalW = 0.0;
            for (Point point : points) {
                totalX += point.getX();
                totalY += point.getY();
                totalZ += point.getZ();
                totalW += point.getW();
            }
            double avgX = totalX / points.size();
            double avgY = totalY / points.size();
            double avgZ = totalZ / points.size();
            double avgW = totalW / points.size();
            return new ClusterCenter(new Point(avgX, avgY, avgZ, avgW));
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, Point, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Point point : values) {
                points.add(new Point(point.getX(), point.getY(), point.getZ(), point.getW()));
            }
            ClusterCenter newCenter = computeNewCenter(points);
            context.write(key, new Text(newCenter.point.getX() + "," + newCenter.point.getY() + ","
                    + newCenter.point.getZ() + "," + newCenter.point.getW()));
        }

        private ClusterCenter computeNewCenter(List<Point> points) {
            double totalX = 0.0;
            double totalY = 0.0;
            double totalZ = 0.0;
            double totalW = 0.0;
            for (Point point : points) {
                totalX += point.getX();
                totalY += point.getY();
                totalZ += point.getZ();
                totalW += point.getW();
            }
            double avgX = totalX / points.size();
            double avgY = totalY / points.size();
            double avgZ = totalZ / points.size();
            double avgW = totalW / points.size();
            return new ClusterCenter(new Point(avgX, avgY, avgZ, avgW));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMeans");

        String inputPath, kSeedData, outputPath;
        inputPath = "C://Users//user//Downloads//Iris.txt";
        kSeedData = "C://Users//user//Downloads//iris_initial_centroids.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_4d_k7";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 7;

        List<ClusterCenter> seedPoints = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(kSeedData))) {
            String line;
            int j = 0;
            while (j < k && (line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                double z = Double.parseDouble(parts[2]);
                double w = Double.parseDouble(parts[3]);
                seedPoints.add(new ClusterCenter(j + 1, new Point(x, y, z, w)));
                j++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        centroidCount = k;
        System.out.println("All seed points" + seedPoints);
        System.out.println("K value: " + centroidCount);

        List<ClusterCenter> selectedCentroids = getRandomCentroids(seedPoints, k);

        centroids = selectedCentroids;
        previousCentroids = new ArrayList<>(centroids);

        job.setJarByClass(KMeans_Iris_Comb.class);
        job.setMapperClass(KMeans_Iris_Comb.KMeansMapper.class);
        job.setCombinerClass(KMeansCombiner.class); // Use the combiner
        job.setReducerClass(KMeans_Iris_Comb.KMeansReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Point.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        int maxIterations = 10;
        boolean converged = false;

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            boolean jobResult = job.waitForCompletion(true);
            if (!jobResult) {
                System.err.println("KMeans Job Failed");
                System.exit(1);
            }

            if (iteration < maxIterations) {
                previousCentroids.clear();
                previousCentroids.addAll(centroids);
                centroids.clear();

                for (int i = 0; i < k; i++) {
                    List<Point> clusterPoints = new ArrayList<>();
                    for (ClusterCenter center : previousCentroids) {
                        if (center.id == i) {
                            clusterPoints.addAll(center.points);
                        }
                    }

                    if (!clusterPoints.isEmpty()) {
                        double totalX = 0.0;
                        double totalY = 0.0;
                        double totalZ = 0.0;
                        double totalW = 0.0;
                        for (Point point : clusterPoints) {
                            totalX += point.getX();
                            totalY += point.getY();
                            totalZ += point.getZ();
                            totalW += point.getW();
                        }

                        double avgX = totalX / clusterPoints.size();
                        double avgY = totalY / clusterPoints.size();
                        double avgZ = totalZ / clusterPoints.size();
                        double avgW = totalW / clusterPoints.size();

                        centroids.add(new ClusterCenter(i, new Point(avgX, avgY, avgZ, avgW)));
                    } else {
                        centroids.add(previousCentroids.get(i));
                    }
                }

                boolean hasConverged = true;
                for (int i = 0; i < k; i++) {
                    double distance = calculateEuclideanDistance(centroids.get(i).point, previousCentroids.get(i).point);
                    if (distance > EARLY_CONVERGENCE_THRESHOLD) {
                        hasConverged = false;
                        break;
                    }
                }

                if (hasConverged) {
                    System.out.println("Converged after " + iteration + " iterations.");
                    converged = true;
                    break;
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
        double dz = p1.getZ() - p2.getZ();
        double dw = p1.getW() - p2.getW();
        return Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    }
}
