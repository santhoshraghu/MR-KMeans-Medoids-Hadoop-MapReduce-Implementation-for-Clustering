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

public class KMedoids_SI {
    public static int medoidCount = 0;
    public static List<Medoid> medoids = new ArrayList<Medoid>();

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

        public Medoid(int id, Point point) {
            this.id = id;
            this.point = point;
        }

        public Medoid(Point point) {
            this.point = point;
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
            double minDistance = Double.MAX_VALUE;
            for (Medoid medoid : medoids) {
                double distance = calculateManhattanDistance(point, medoid.point);
                if (distance < minDistance) {
                    nearestMedoid = medoid;
                    minDistance = distance;
                }
            }
            return nearestMedoid;
        }

        private double calculateManhattanDistance(Point p1, Point p2) {
            return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
        }
    }

    public static class KMedoidsReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Text value : values) {
                String[] parts = value.toString().split(":");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                points.add(new Point(x, y));
            }
            Medoid currentMedoid = null;
            double minTotalDistance = Double.MAX_VALUE;
            for (Medoid medoid : medoids) {
                double totalDistance = 0.0;
                for (Point point : points) {
                    totalDistance += calculateManhattanDistance(point, medoid.point);
                }
                if (totalDistance < minTotalDistance) {
                    currentMedoid = new Medoid(medoid.id, medoid.point);
                    minTotalDistance = totalDistance;
                }
            }
            context.write(key, new Text(currentMedoid.point.getX() + "," + currentMedoid.point.getY()));
        }

        private double calculateManhattanDistance(Point p1, Point p2) {
            return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMedoids");

        String inputPath, medoidInputPath, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        medoidInputPath = "C://Users//user//Downloads//seed_points.txt"; // File containing initial medoid points
        outputPath = "C://Users//user//Downloads//Output_Task_3a_k7";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 7; // Set the desired value of k (number of medoids)


        // Read initial medoids from a file
        // Read all points from the input file and store them in a list
        List<Medoid> allMedoids = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(inputPath))) {
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

// Set the medoidCount to k
        medoidCount = k;

        List<Medoid> selectedMedoids = getRandomMedoids(allMedoids, k);


        // Initialize your medoids list with the selected random k medoids
        medoids = selectedMedoids;

        job.setJarByClass(KMedoids_SI.class);
        job.setMapperClass(KMedoidsMapper.class);
        job.setReducerClass(KMedoidsReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        int r = 1;

        for (int i = 0; i < r; i++) {
            job.waitForCompletion(true);
        }
    }

    private static List<Medoid> getRandomMedoids(List<Medoid> seedMedoids, int k) {
        List<Medoid> selectedMedoids = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis());

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
}
