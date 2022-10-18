using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ml_pred_ConsoleApp1.Schemas
{
    internal class LoadForecasting
    {
        [ColumnName("Score")]
        public float DAYF { get; set; }
    }
}
