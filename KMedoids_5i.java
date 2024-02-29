import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FSDataOutputStream;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMedoids_5i {
    public static int medoidCount = 0;
    public static List<Medoid> medoids = new ArrayList<Medoid>();
    public static List<Medoid> previousMedoids = new ArrayList<Medoid>();
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

        public Medoid() {
        }

        public Medoid(int id, Point point) {
            this.id = id;
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

    public static class KMedoidsMapper extends Mapper<LongWritable, Text, IntWritable, Point> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] pointData = value.toString().split(",");
            double x = Double.parseDouble(pointData[0]);
            double y = Double.parseDouble(pointData[1]);
            Point point = new Point(x, y);
            Medoid nearestMedoid = findNearestMedoid(point);
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
            // The K-Medoids algorithm doesn't require finding new medoids in the combiner.
            // Here, we simply pass the original medoids as is.
            for (Point point : values) {
                context.write(key, point);
            }
        }
    }

    public static class KMedoidsReducer extends Reducer<IntWritable, Point, IntWritable, Point> {
        @Override
        public void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            // The K-Medoids algorithm doesn't require finding new medoids in the reducer.
            // Here, we simply pass the original medoids as is.
            for (Point point : values) {
                context.write(key, point);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "KMedoids");

        String inputPath, kSeedData, outputPath;
        inputPath = "C://Users//user//Downloads//data_points.txt";
        kSeedData = "C://Users//user//Downloads//seed_points.txt";
        outputPath = "C://Users//user//Downloads//Output_Task_4e_k33";

        conf.set("mapreduce.output.textoutputformat.separator", ",");

        FileSystem fs = FileSystem.get(conf);
        Path outPath = new Path(outputPath);
        if (fs.exists(outPath)) {
            fs.delete(outPath, true);
        }

        int k = 3;

        List<Medoid> seedMedoids = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(kSeedData))) {
            String line;
            int j = 0;
            while (j < k && (line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                seedMedoids.add(new Medoid(j + 1, new Point(x, y)));
                j++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        medoidCount = k;
        System.out.println("All seed medoids" + seedMedoids);
        System.out.println("K value: " + medoidCount);

        List<Medoid> selectedMedoids = getRandomMedoids(seedMedoids, k);

        medoids = selectedMedoids;
        previousMedoids = new ArrayList<>(medoids);

        job.setJarByClass(KMedoids_5i.class);
        job.setMapperClass(KMedoids_5i.KMedoidsMapper.class);
        job.setCombinerClass(KMedoidsCombiner.class);
        job.setReducerClass(KMedoids_5i.KMedoidsReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Point.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Point.class);

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
                            clusterPoints.add(medoid.point);
                        }
                    }

                    if (!clusterPoints.isEmpty()) {
                        Point newMedoid = findNewMedoid(clusterPoints);
                        medoids.add(new Medoid(i, newMedoid));
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

        if (converged) {
            System.out.println("Converged. Final Medoids:");

            // Write convergence and final medoid information to an output file
            Path medoidOutputPath = new Path(outputPath, "medoids.txt");
            try (FSDataOutputStream out = fs.create(medoidOutputPath)) {
                String finalMedoidInfo = "Converged. Final Medoids:\n";
                out.write(finalMedoidInfo.getBytes());

                for (Medoid medoid : medoids) {
                    String outputLine = medoid.point.getX() + "," + medoid.point.getY() + "\n";
                    out.write(outputLine.getBytes());
                }
            }

            System.out.println("Final medoids and convergence information have been written to: " + medoidOutputPath);
        }
    }

    private static List<Medoid> getRandomMedoids(List<Medoid> seedMedoids, int k) {
        List<Medoid> selectedMedoids = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis());

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

    private static Point findNewMedoid(List<Point> clusterPoints) {
        Point newMedoid = clusterPoints.get(0); // Initialize with the first point
        double minTotalDistance = calculateTotalDistance(clusterPoints, newMedoid);

        for (Point point : clusterPoints) {
            double totalDistance = calculateTotalDistance(clusterPoints, point);
            if (totalDistance < minTotalDistance) {
                minTotalDistance = totalDistance;
                newMedoid = point;
            }
        }

        return newMedoid;
    }

    private static double calculateTotalDistance(List<Point> points, Point medoid) {
        double totalDistance = 0.0;
        for (Point point : points) {
            totalDistance += calculateManhattanDistance(point, medoid);
        }
        return totalDistance;
    }

    private static double calculateManhattanDistance(Point p1, Point p2) {
        return Math.abs(p1.getX() - p2.getX()) + Math.abs(p1.getY() - p2.getY());
    }
}
