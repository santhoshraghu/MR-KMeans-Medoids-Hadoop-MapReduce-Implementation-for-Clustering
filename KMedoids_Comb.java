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

public class KMedoids_Comb {
    public static int medoidCount = 0;
    public static List<Medoid> medoids = new ArrayList<Medoid>();
    public static List<Medoid> previousMedoids = new ArrayList<Medoid>();
    public static List<Point> k_seeds = new ArrayList<Point>();

    // Define your early convergence threshold (e.g., the minimum change in medoids)
    public static final double EARLY_CONVERGENCE_THRESHOLD = 0.001;

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

        public void addPoint(Point point) {
            points.add(point);
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            point.write(dataOutput);
            dataOutput.writeInt(id);
            dataOutput.writeInt(points.size());
            for (Point p : points) {
                p.write(dataOutput);
            }
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            point.readFields(dataInput);
            id = dataInput.readInt();
            int numPoints = dataInput.readInt();
            points.clear();
            for (int i = 0; i < numPoints; i++) {
                Point p = new Point();
                p.readFields(dataInput);
                points.add(p);
            }
        }
    }

    public static class KMedoidsMapper extends Mapper<LongWritable, Text, IntWritable, Point> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] pointData = value.toString().split(",");
            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);
            Point point = new Point(x, y);
            Medoid nearestMedoid = findNearestMedoid(point);
            nearestMedoid.addPoint(point);
            context.write(new IntWritable(nearestMedoid.id), point);
        }

        private Medoid findNearestMedoid(Point point) {
            Medoid nearestMedoid = null;
            double temp = Double.MAX_VALUE;
            for (Medoid medoid : medoids) {
                Point medoidPoint = medoid.getPoint();
                double distance = calculateManhattanDistance(point, medoidPoint);
                if (distance < temp) {
                    nearestMedoid = medoid;
                    temp = distance;
                }
            }
            return nearestMedoid;
        }

        private double calculateManhattanDistance(Point p1, Point p2) {
            return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
        }
    }

    public static class KMedoidsCombiner extends Reducer<IntWritable, Point, IntWritable, Point> {
        @Override
        public void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Point point : values) {
                points.add(new Point(point.getX(), point.getY()));
            }
            Medoid newMedoid = computeNewMedoid(points);
            context.write(key, newMedoid.getPoint());
        }
    }

    public static class KMedoidsReducer extends Reducer<IntWritable, Point, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Point point : values) {
                points.add(new Point(point.getX(), point.getY()));
            }
            Medoid newMedoid = computeNewMedoid(points);
            context.write(key, new Text(newMedoid.point.getX() + "," + newMedoid.point.getY()));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMedoids");

        String inputPath, medoidInputPath, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        medoidInputPath = "C://Users//user//Downloads//seed_points.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_3d_k7";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 7;

        List<Medoid> allMedoids = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(medoidInputPath))) {
            String line;
            int id = 1;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                allMedoids.add(new Medoid(id, new Point(x, y)));
                id++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        medoidCount = k;
        System.out.println("All seed medoids" + allMedoids);
        System.out.println("K value: " + medoidCount);

        List<Medoid> selectedMedoids = getRandomMedoids(allMedoids, k);

        medoids = selectedMedoids;
        previousMedoids = new ArrayList<>(medoids);

        job.setJarByClass(KMedoids_Comb.class);
        job.setMapperClass(KMedoidsMapper.class);
        job.setCombinerClass(KMedoidsCombiner.class); // Use the combiner
        job.setReducerClass(KMedoidsReducer.class);

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
                System.err.println("KMedoids Job Failed");
                System.exit(1);
            }

            if (iteration < maxIterations) {
                previousMedoids.clear();
                previousMedoids.addAll(medoids);
                medoids.clear();

                for (int i = 0; i < k; i++) {
                    List<Point> clusterPoints = new ArrayList<>();
                    for (Medoid medoid : previousMedoids) {
                        if (medoid.id == i) {
                            clusterPoints.addAll(medoid.points);
                        }
                    }

                    if (!clusterPoints.isEmpty()) {
                        Medoid newMedoid = computeNewMedoid(clusterPoints);
                        medoids.add(new Medoid(i, newMedoid.getPoint()));
                    } else {
                        medoids.add(previousMedoids.get(i));
                    }
                }

                boolean hasConverged = true;
                for (int i = 0; i < k; i++) {
                    double distance = calculateManhattanDistance(medoids.get(i).point, previousMedoids.get(i).point);
                    if (distance > EARLY_CONVERGENCE_THRESHOLD) {
                        hasConverged = false;
                        break;
                    }
                }

                if (hasConverged) {
                    System.out.println("Converged after " + iteration+3 + " iterations.");
                    converged = true;
                    break;
                }
            }
        }
    }

    private static List<Medoid> getRandomMedoids(List<Medoid> seedMedoids, int k) {
        List<Medoid> selectedMedoids = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis()); // Use current time as the seed

        if (k > seedMedoids.size()) {
            k = seedMedoids.size();
        }

        while (selectedMedoids.size() < k) {
            int randomIndex = rand.nextInt(seedMedoids.size());
            Medoid randomMedoid = seedMedoids.get(randomIndex);

            if (!selectedMedoids.contains(randomMedoid)) {
                selectedMedoids.add(randomMedoid);
            }
        }

        return selectedMedoids;
    }

    private static double calculateManhattanDistance(Point p1, Point p2) {
        return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
    }

    private static Medoid computeNewMedoid(List<Point> points) {
        Medoid nearestMedoid = null;
        double minTotalDistance = Double.MAX_VALUE;

        for (Point candidateMedoid : points) {
            double totalDistance = 0.0;
            for (Point point : points) {
                totalDistance += calculateManhattanDistance(candidateMedoid, point);
            }
            if (totalDistance < minTotalDistance) {
                nearestMedoid = new Medoid(candidateMedoid);
                minTotalDistance = totalDistance;
            }
        }
        return nearestMedoid;
    }
}
