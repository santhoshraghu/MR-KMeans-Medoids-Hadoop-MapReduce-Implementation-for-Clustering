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

public class KMedoids_5ii {

    public static int medoidCount = 0;
    public static List<Medoid> medoids = new ArrayList<Medoid>();
    public static List<Medoid> previousMedoids = new ArrayList<Medoid>();
    public static List<Point> k_seeds = new ArrayList<Point>();
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
        public List<Point> points;  // Store associated data points

        public Medoid() {
            this.points = new ArrayList<Point>();
        }

        public Medoid(int id, Point point) {
            this.id = id;
            this.point = point;
            this.points = new ArrayList<Point>();
        }

        public Point getPoint() {
            return point;
        }

        @Override
        public void write(DataOutput dataOutput) throws IOException {
            point.write(dataOutput);
            dataOutput.writeInt(id);
        }

        @Override
        public void readFields(DataInput dataInput) throws IOException {
            point.readFields(dataInput);
            id = dataInput.readInt();
        }

        public void addPoint(Point point) {
            points.add(point);
        }
    }

    public static class KMedoidsMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] pointData = value.toString().split(",");
            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);
            Point point = new Point(x, y);
            Medoid nearestMedoid = findNearestMedoid(point);
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
                    temp = distance;
                }
            }
            return nearestMedoid;
        }

        private double calculateManhattanDistance(Point p1, Point p2) {
            return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
        }
    }

    public static class KMedoidsCombiner extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                context.write(key, value);
            }
        }
    }

    public static class KMedoidsReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            List<KMeans_5ii.Point> points = new ArrayList<>();

            for (Text value : values) {
                String[] parts = value.toString().split(":");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                points.add(new KMeans_5ii.Point(x, y));
            }

            KMeans_5ii.ClusterCenter newCenter = computeNewCenter(points);

            // Emit the cluster center with an identifier to distinguish it from data points
            context.write(new IntWritable(key.get()), new Text("M:" + newCenter.point.getX() + "," + newCenter.point.getY()));

            // Emit the data points associated with this cluster center
            for (KMeans_5ii.Point point : points) {
                context.write(new IntWritable(key.get()), new Text("D:" + point.getX() + "," + point.getY()));
            }
        }



        private KMeans_5ii.ClusterCenter computeNewCenter(List<KMeans_5ii.Point> points) {
            double totalX = 0.0;
            double totalY = 0.0;
            for (KMeans_5ii.Point point : points) {
                totalX += point.getX();
                totalY += point.getY();
            }
            double avgX = totalX / points.size();
            double avgY = totalY / points.size();
            return new KMeans_5ii.ClusterCenter(new KMeans_5ii.Point(avgX, avgY));
        }
    }





    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMedoids");

        String inputPath, medoidSeedData, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        medoidSeedData = "C://Users//user//Downloads//seed_points.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_3f_k7";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 7;

        List<KMedoids_5ii.Medoid> seedMedoids = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(medoidSeedData))) {
            String line;
            int j = 0;
            while (j < k && (line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                seedMedoids.add(new KMedoids_5ii.Medoid(j + 1, new KMedoids_5ii.Point(x, y)));
                j++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        medoidCount = k;
        System.out.println("All seed medoids" + seedMedoids);
        System.out.println("K value: " + medoidCount);

        medoids = seedMedoids;
        previousMedoids = new ArrayList<>(medoids);

        job.setJarByClass(KMedoids_5ii.class);
        job.setMapperClass(KMedoidsMapper.class);
        job.setCombinerClass(KMedoidsCombiner.class);
        job.setReducerClass(KMedoidsReducer.class);

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
                        double totalDistance = calculateTotalDistance(clusterPoints, previousMedoids.get(i).point);
                        Medoid newMedoid = new Medoid(i + 1, previousMedoids.get(i).point);
                        for (Point point : clusterPoints) {
                            double newDistance = calculateTotalDistance(clusterPoints, point);
                            if (newDistance < totalDistance) {
                                totalDistance = newDistance;
                                newMedoid = new Medoid(i + 1, point);
                            }
                        }
                        medoids.add(newMedoid);
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
                    System.out.println("Converged after " + iteration+2 + " iterations.");
                    converged = true;
                    break;
                }
            }
        }
    }

    private static double calculateManhattanDistance(Point p1, Point p2) {
        return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
    }

    private static double calculateTotalDistance(List<Point> points, Point medoid) {
        double totalDistance = 0.0;
        for (Point point : points) {
            totalDistance += calculateManhattanDistance(point, medoid);
        }
        return totalDistance;
    }
}
