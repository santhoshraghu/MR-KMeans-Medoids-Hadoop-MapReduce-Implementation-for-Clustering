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

public class KMedoids_BMI {
    public static int medoidCount = 0;
    public static List<Medoid> medoids = new ArrayList<Medoid>();
    public static List<Medoid> previousMedoids = new ArrayList<Medoid>();
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

    public static class Medoid implements Writable {
        public Point point = new Point(0, 0);
        public int id;
        public int count = 0;

        public List<Point> points;

        public Medoid() {
            this.points = new ArrayList<Point>();
        }

        public Medoid(int id, Point point) {
            this.points = new ArrayList<Point>();
            this.id = id;
            this.point = point;
        }

        public Medoid(Point point) {
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

    public static class KMedoidsMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Mapper logic: Assign data points to the nearest medoid
            String[] pointData = value.toString().split(",");
            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);
            Point point = new Point(x, y);
            Medoid nearestMedoid = findNearestMedoid(point);
            nearestMedoid.addPoint(point);
            context.write(new IntWritable(nearestMedoid.id), new Text(x + ":" + y));
        }

        private Medoid findNearestMedoid(Point point) {
            Medoid nearestMedoid = null;
            double temp = Double.MAX_VALUE;
            for (Medoid medoid : medoids) {
                Point medoidPoint = medoid.getPoint();
                double distance = calculateManhattanDistance(point, medoidPoint);
                if (distance < temp) {
                    nearestMedoid = medoid;
                }
                temp = distance;
            }
            return nearestMedoid;
        }

        private double calculateManhattanDistance(Point p1, Point p2) {
            return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
        }
    }

    public static class KMedoidsReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Reducer logic: Recalculate medoids based on assigned data points
            List<Point> points = new ArrayList<>();
            for (Text value : values) {
                String[] parts = value.toString().split(":");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                points.add(new Point(x, y));
            }
            Medoid newMedoid = computeNewMedoid(points);
            context.write(key, new Text(newMedoid.point.getX() + "," + newMedoid.point.getY()));
        }

        private Medoid computeNewMedoid(List<Point> points) {
            Medoid newMedoid = null;
            double minTotalDistance = Double.MAX_VALUE;
            for (Medoid medoid : medoids) {
                double totalDistance = 0.0;
                for (Point point : points) {
                    totalDistance += calculateManhattanDistance(point, medoid.point);
                }
                if (totalDistance < minTotalDistance) {
                    newMedoid = new Medoid(medoid.id, medoid.point);
                    minTotalDistance = totalDistance;
                }
            }
            return newMedoid;
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMedoids");

        String inputPath, medoidInputPath, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        medoidInputPath = "C://Users//user//Downloads//seed_points.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_3b_k777";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 7; // Set the desired value of k

        // Read all points from the seed points file and store them in a list
        List<KMedoids_BMI.Medoid> allMedoids = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(inputPath))) {
            String line;
            int id = 1;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                allMedoids.add(new KMedoids_BMI.Medoid(id, new KMedoids_BMI.Point(x, y)));
                id++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Set the medoidCount to k
        medoidCount = k;

        List<Medoid> selectedMedoids = getRandomMedoids(allMedoids, k);

        // Initialize your medoids list with the selected random k medoids
        medoids = selectedMedoids;
        previousMedoids = new ArrayList<>(medoids); // Store the previous medoids

        job.setJarByClass(KMedoids_BMI.class);
        job.setMapperClass(KMedoidsMapper.class);
        job.setReducerClass(KMedoidsReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        int maxIterations = 10;
        boolean converged = false; // To check if medoids have not changed for two consecutive iterations

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            boolean jobResult = job.waitForCompletion(true);
            if (!jobResult) {
                System.err.println("KMedoids Job Failed");
                System.exit(1);
            }

            // Update medoids for the next iteration
            if (iteration < maxIterations) {
                previousMedoids.clear();
                previousMedoids.addAll(medoids);
                medoids.clear();

                // Calculate new medoids based on the reducer's output
                for (int i = 0; i < k; i++) {
                    List<Point> clusterPoints = new ArrayList<>();
                    for (Medoid medoid : previousMedoids) { // Use previousMedoids here
                        if (medoid.id == i) {
                            clusterPoints.addAll(medoid.points);
                        }
                    }

                    if (!clusterPoints.isEmpty()) {
                        // Find the point in the cluster that minimizes the total distance
                        Medoid newMedoid = null;
                        double minTotalDistance = Double.MAX_VALUE;
                        for (Point point : clusterPoints) {
                            Medoid tempMedoid = new Medoid(i, point);
                            double totalDistance = 0.0;
                            for (Point p : clusterPoints) {
                                totalDistance += calculateManhattanDistance(point, p);
                            }
                            if (totalDistance < minTotalDistance) {
                                newMedoid = tempMedoid;
                                minTotalDistance = totalDistance;
                            }
                        }

                        medoids.add(newMedoid);
                    } else {
                        // If the cluster is empty, keep the previous medoid
                        medoids.add(previousMedoids.get(i));
                    }
                }

                // Check for convergence
                boolean hasConverged = true;
                for (int i = 0; i < k; i++) {
                    double distance = calculateManhattanDistance(medoids.get(i).point, previousMedoids.get(i).point);
                    if (distance > 0.0) {
                        hasConverged = false;
                        break;
                    }
                }

                if (hasConverged) {
                    System.out.println("Converged after " + iteration+2 + " iterations.");
                    converged = true; // Set the converged flag if medoids have converged
                    break; // Terminate the loop
                }
            }
        }
    }

    private static List<Medoid> getRandomMedoids(List<Medoid> seedMedoids, int k) {
        List<Medoid> selectedMedoids = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis()); // Use current time as the seed

        // Ensure k is not greater than the number of available seed medoids
        if (k > seedMedoids.size()) {
            k = seedMedoids.size();
        }

        while (selectedMedoids.size() < k) {
            int randomIndex = rand.nextInt(seedMedoids.size());
            Medoid randomMedoid = seedMedoids.get(randomIndex);

            // Avoid selecting the same medoid multiple times
            if (!selectedMedoids.contains(randomMedoid)) {
                selectedMedoids.add(randomMedoid);
            }
        }

        return selectedMedoids;
    }

    private static double calculateManhattanDistance(Point p1, Point p2) {
        return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
    }
}
