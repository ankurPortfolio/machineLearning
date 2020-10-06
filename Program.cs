using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;

namespace HouseRatePrediction
{

    public class HousingData
    {
        [LoadColumn(0)]
        public float Size { get; set; }

        [LoadColumn(1, 3)]
        [VectorType(3)]
        public float[] HistoricalPrices { get; set; }

        [LoadColumn(4)]
        [ColumnName("Label")]
        public float CurrentPrice { get; set; }
    }

    class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {

            HousingData[] housingData = new HousingData[]
            {
                new HousingData
                {
                    Size = 600f,
                    HistoricalPrices = new float[] { 100000f, 125000f, 122000f },
                    CurrentPrice = 170000f
                },
                new HousingData
                {
                    Size = 1000f,
                    HistoricalPrices = new float[] { 200000f, 250000f, 230000f },
                    CurrentPrice = 225000f
                },
                new HousingData
                {
                    Size = 1000f,
                    HistoricalPrices = new float[] { 126000f, 130000f, 200000f },
                    CurrentPrice = 195000f
                },
                new HousingData
                {
                    Size = 1000f,
                    HistoricalPrices = new float[] { 126000f, 300000f, 200000f },
                    CurrentPrice = 505000f
                }
            };

            // Create MLContext
            MLContext mlContext = new MLContext();

            // Load Data
            IDataView data = mlContext.Data.LoadFromEnumerable<HousingData>(housingData);

            // Define data preparation estimator
            EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> pipelineEstimator =
                mlContext.Transforms.Concatenate("Features", new string[] { "Size", "HistoricalPrices" })
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                    .Append(mlContext.Regression.Trainers.Sdca());

            // Train model
            ITransformer trainedModel = pipelineEstimator.Fit(data);

            // Save model
            mlContext.Model.Save(trainedModel, data.Schema, @"C:\Users\ankur.agarwal\Desktop\Training\model.zip");



            //---------------------------Make Predictions---------------------------------//


            //Create MLContext
            MLContext mlContext1 = new MLContext();

            // Load Trained Model
            DataViewSchema predictionPipelineSchema;
            ITransformer predictionPipeline = mlContext1.Model.Load(@"C:\Users\ankur.agarwal\Desktop\Training\model.zip", out predictionPipelineSchema);

            // Create PredictionEngines
            PredictionEngine<HousingData, HousingPrediction> predictionEngine = mlContext1.Model.CreatePredictionEngine<HousingData, HousingPrediction>(predictionPipeline);

            // Input Data
            HousingData inputData = new HousingData
            {
                Size = 900f,
                HistoricalPrices = new float[] { 10000f, 190000f, 220000f }
            };

            // Get Prediction
            HousingPrediction prediction = predictionEngine.Predict(inputData);




            Console.WriteLine("Hello World!");
        }
    }
}
