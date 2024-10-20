using System;
using Microsoft.ML;
using Microsoft.ML.Data;

class CustomerSegmentationExample
{
  static void Main(string[] args)
  {
    var mlContext = new MLContext();
    
    // Data from csv
    IDataView dataView = mlContext.Data.LoadFromTextFile<CustomerData>("customerData.csv", hasHeader: true, separatorChar: ',');

    // pipeline 
    var pipeline = mlContext.Transforms.Concatenate("Features", nameof(CustomerData.Age), nameof(CustomerData.Salary))
        .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3));
   
    // train 
    var model = pipeline.Fit(dataView);

    // save the model
    mlContext.Model.Save(model, dataView.Schema, "customerClusteringModel.zip");

    // predict cluster
    var predictions = model.Transform(dataView);
    var clusters = mlContext.Data.CreateEnumerable<ClusterPredictionWithData>(predictions, reuseRowObject: false);



  }
}
public class CustomerData
{
  public float Age { get; set; }
  public float Salary { get; set; }
}

public class ClusterPrediction
{
  [ColumnName("PredictedLabel")]
  public uint PredictedClusterId { get; set; }
}

