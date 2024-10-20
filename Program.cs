using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

class CustomerSegmentationExample
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load the data from CSV
        string filePath = "customers.csv";  
        IDataView dataView = mlContext.Data.LoadFromTextFile<CustomerData>(filePath, hasHeader: true, separatorChar: ',');

        // pipeline 
        var pipeline = mlContext.Transforms.Concatenate("Features", nameof(CustomerData.Age), nameof(CustomerData.Salary))
            .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3));

        // Train the model
        var model = pipeline.Fit(dataView);

        // Predict clusters
        var predictions = model.Transform(dataView);

        //output schema 
        var schema = predictions.Schema;
        Console.WriteLine("Columns in the predictions DataView:");
        foreach (var column in schema)
        {
            Console.WriteLine($"{column.Name}");
        }

        // data and output predictions
        var clusters = mlContext.Data.CreateEnumerable<ClusterPredictionWithData>(predictions, reuseRowObject: false);

        // Output the clustering results
        Console.WriteLine("Customer Clustering Results:");
        foreach (var customer in clusters)
        {
            Console.WriteLine($"Age: {customer.Age}, Salary: {customer.Salary}, Cluster: {customer.PredictedClusterId}");
        }
    }
}

//classes
public class CustomerData
{
    [LoadColumn(0)] 
    public float Age { get; set; }

    [LoadColumn(1)]  
    public float Salary { get; set; }
}

public class ClusterPredictionWithData : CustomerData
{
    [ColumnName("PredictedLabel")]  
    public uint PredictedClusterId { get; set; }
}
