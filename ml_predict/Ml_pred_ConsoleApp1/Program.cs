using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

using System.Data.Common;

using Microsoft.ML;
using Microsoft.ML.Data;

//using PLplot;

using Ml_pred_ConsoleApp1.Schemas;


namespace Ml_pred_ConsoleApp1
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../../Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/load_traindata.csv";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/load_testdata.csv";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static string BaseModelsRelativePath = @"../../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/LoadModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {

            //Create ML Context with seed for repeatable/deterministic results
            MLContext mlContext = new MLContext(seed: 0);

            // Create, Train, Evaluate and Save a model
            BuildTrainEvaluateAndSaveModel(mlContext);

            // Make a single test prediction loding the model from .ZIP file
            TestSinglePrediction(mlContext);

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();

        }


        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext context)
        {
            // STEP 1: Common data loading configuration
            IDataView baseTrainingDataView = context.Data.LoadFromTextFile<LoadParameters>(TrainDataPath, hasHeader: true, separatorChar: ';');
            IDataView testDataView = context.Data.LoadFromTextFile<LoadParameters>(TestDataPath, hasHeader: true, separatorChar: ';');

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(LoadParameters.DAYF))
                            .Append(context.Transforms.NormalizeMinMax(outputColumnName: nameof(LoadParameters.DAY7))
                            .Append(context.Transforms.NormalizeMinMax(outputColumnName: nameof(LoadParameters.DAY2)))
                            .Append(context.Transforms.Concatenate("Features", nameof(LoadParameters.DAY7), nameof(LoadParameters.DAY2))));

            // (OPTIONAL) Peek data (such as 5 records) in training DataView after applying the ProcessPipeline's transformations into "Features" 
            //ConsoleHelper.PeekDataViewInConsole(context, baseTrainingDataView, dataProcessPipeline, 5);
            //ConsoleHelper.PeekVectorColumnDataInConsole(context, "Features", baseTrainingDataView, dataProcessPipeline, 5);

            // STEP 3: Set the training algorithm, then create and config the modelBuilder
            var trainer = context.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);


            // STEP 4: Train model, then create and config the modelBuilder
            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(baseTrainingDataView);

            // STEP 5: Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");

            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Common.ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);
            //private static string ModelPath = Path.GetFullPath(@"D:\CodeExercises\ml_predict\ml_predict\ml_pred.zip");
            // STEP 6: Save/persist the trained model to a .ZIP file
            context.Model.Save(trainedModel, baseTrainingDataView.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;
        }
        private static void TestSinglePrediction(MLContext mlContext)
        {
            //Sample: 
            //DAY7,DAY2,DAYF
            //VTS,1,1,1140,3.75,CRD,15.5

            var LoadSample = new LoadParameters()
            {
                DAY7 = 1,
                DAY2 = 2,
                DAYF = 1, // To predict. Actual/Observed = 15.5
            };

            ///
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<LoadParameters, LoadConsumptionForecasting>(trainedModel);

            //Score
            var resultprediction = predEngine.Predict(LoadSample);
            ///

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {resultprediction.DAYF:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }


        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public class loadCsvReader
        {
            public IEnumerable<LoadParameters> GetDataFromCsv(string dataLocation, int numMaxRecords)
            {
                IEnumerable<LoadParameters> records =
                    File.ReadAllLines(dataLocation)
                    .Skip(1)
                    .Select(x => x.Split(';'))
                    .Select(x => new LoadParameters()
                    {
                        DAY7 = float.Parse(x[0], CultureInfo.InvariantCulture),
                        DAY2 = float.Parse(x[1], CultureInfo.InvariantCulture),
                        DAYF = float.Parse(x[2], CultureInfo.InvariantCulture)
                    })
                    .Take<LoadParameters>(numMaxRecords);

                return records;
            }
        }
    }
}

