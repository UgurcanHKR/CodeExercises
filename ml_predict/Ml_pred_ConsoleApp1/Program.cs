using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

//using Common;

using Microsoft.ML;
using Microsoft.ML.Data;

//using PLplot;

using Ml_pred_ConsoleApp1.Schemas;


namespace Ml_pred_ConsoleApp1
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        /*
        private static string BaseDatasetsRelativePath = @"Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}\\load_traindata.csv";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}\\load_testdata.csv";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static string BaseModelsRelativePath = @"MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}\\LoadModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);
        */
        private static string TrainDataPath = "D:\\CodeExercises\\ml_predict\\Ml_pred_ConsoleApp1\\Data\\load_traindata.csv";
        private static string TestDataPath = "D:\\CodeExercises\\ml_predict\\Ml_pred_ConsoleApp1\\Data\\load_testdata.csv";
        private static string ModelPath = "D:\\CodeExercises\\ml_predict\\Ml_pred_ConsoleApp1\\MLmodels\\ml_pred.zip";

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

            //Common.ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);
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
                DAY7 = 33092.96F,
                DAY2 = 32355.94F,
                DAYF = 35516.13F, // To predict. 
            };

            ///
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<LoadParameters, LoadForecasting>(trainedModel);

            //Score
            var resultprediction = predEngine.Predict(LoadSample);
            ///

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Forecasted value: {resultprediction.DAYF:0.####}, Actual value: 35516.13");
            Console.WriteLine($"**********************************************************************");
        }
        /*
        private static void PlotRegressionChart(MLContext mlContext,
                                        string testDataSetPath,
                                        int numberOfRecordsToRead,
                                        string[] args)
        {
            ITransformer trainedModel;
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            // Create prediction engine related to the loaded trained model
            var predFunction = mlContext.Model.CreatePredictionEngine<LoadParameters, LoadForecasting>(trainedModel);

            string chartFileName = "";
            using (var pl = new PLStream())
            {
                // use SVG backend and write to SineWaves.svg in current directory
                if (args.Length == 1 && args[0] == "svg")
                {
                    pl.sdev("svg");
                    chartFileName = "LoadRegressionDistribution.svg";
                    pl.sfnam(chartFileName);
                }
                else
                {
                    pl.sdev("pngcairo");
                    chartFileName = "LoadRegressionDistribution.png";
                    pl.sfnam(chartFileName);
                }

                // use white background with black foreground
                pl.spal0("cmap0_alternate.pal");

                // Initialize plplot
                pl.init();

                // set axis limits
                const int xMinLimit = 0;
                const int xMaxLimit = 35; //Rides larger than $35 are not shown in the chart
                const int yMinLimit = 0;
                const int yMaxLimit = 35;  //Rides larger than $35 are not shown in the chart
                pl.env(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

                // Set scaling for mail title text 125% size of default
                pl.schr(0, 1.25);

                // The main title
                pl.lab("Measured", "Predicted", "Distribution of Taxi Fare Prediction");

                // plot using different colors
                // see http://plplot.sourceforge.net/examples.php?demo=02 for palette indices
                pl.col0(1);

                int totalNumber = numberOfRecordsToRead;
                var testData = new loadCsvReader().GetDataFromCsv(testDataSetPath, totalNumber).ToList();

                //This code is the symbol to paint
                char code = (char)9;

                // plot using other color
                //pl.col0(9); //Light Green
                //pl.col0(4); //Red
                pl.col0(2); //Blue

                double yTotal = 0;
                double xTotal = 0;
                double xyMultiTotal = 0;
                double xSquareTotal = 0;

                for (int i = 0; i < testData.Count; i++)
                {
                    var x = new double[1];
                    var y = new double[1];

                    //Make Prediction
                    var FarePrediction = predFunction.Predict(testData[i]);

                    x[0] = testData[i].FareAmount;
                    y[0] = FarePrediction.FareAmount;

                    //Paint a dot
                    pl.poin(x, y, code);

                    xTotal += x[0];
                    yTotal += y[0];

                    double multi = x[0] * y[0];
                    xyMultiTotal += multi;

                    double xSquare = x[0] * x[0];
                    xSquareTotal += xSquare;

                    double ySquare = y[0] * y[0];

                    Console.WriteLine($"-------------------------------------------------");
                    Console.WriteLine($"Predicted : {FarePrediction.FareAmount}");
                    Console.WriteLine($"Actual:    {testData[i].FareAmount}");
                    Console.WriteLine($"-------------------------------------------------");
                }

                // Regression Line calculation explanation:
                // https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/regression-line-example

                double minY = yTotal / totalNumber;
                double minX = xTotal / totalNumber;
                double minXY = xyMultiTotal / totalNumber;
                double minXsquare = xSquareTotal / totalNumber;

                double m = ((minX * minY) - minXY) / ((minX * minX) - minXsquare);

                double b = minY - (m * minX);

                //Generic function for Y for the regression line
                // y = (m * x) + b;

                double x1 = 1;
                //Function for Y1 in the line
                double y1 = (m * x1) + b;

                double x2 = 39;
                //Function for Y2 in the line
                double y2 = (m * x2) + b;

                var xArray = new double[2];
                var yArray = new double[2];
                xArray[0] = x1;
                yArray[0] = y1;
                xArray[1] = x2;
                yArray[1] = y2;

                pl.col0(4);
                pl.line(xArray, yArray);

                // end page (writes output to disk)
                pl.eop();

                // output version of PLplot
                pl.gver(out var verText);
                Console.WriteLine("PLplot version " + verText);

            } // the pl object is disposed here

            // Open Chart File In Microsoft Photos App (Or default app, like browser for .svg)

            Console.WriteLine("Showing chart...");
            var p = new Process();
            string chartFileNamePath = @".\" + chartFileName;
            p.StartInfo = new ProcessStartInfo(chartFileNamePath)
            {
                UseShellExecute = true
            };
            p.Start();
        }
        */
        public static string GetAbsolutePath(string relativePath)
        {
            var a = typeof(Program);
            var b = a.Assembly;
            var c = b.Location;
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

