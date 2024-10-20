using Microsoft.ML;
using Microsoft.ML.Data;

class CustomerSegmentationExample
{
  static void Main(string[] args)
  {
    var mlContext = new MLContext();
    
    IDataView dataView = mlContext.Data.LoadFromTextFile<CustomerData>("customerData.csv", hasHeader: true, separatorChar: ',');

    var pipeline = mlContext.Transforms.Concatenate("Features", nameof(CustomerData.Age), nameof(CustomerData.Salary))
        .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3));

  }
}
